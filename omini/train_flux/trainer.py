import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
import time

from typing import List

import prodigyopt

from ..pipeline.flux_omini import transformer_forward, encode_images


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, prompts = batch["image"], batch["description"]
        image_latent_mask = batch.get("image_latent_mask", None)

        # Get the conditions and position deltas from the batch
        conditions, position_deltas, position_scales, latent_masks = [], [], [], []
        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)

            # Prepare text input
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]

            # Prepare conditions
            condition_latents, condition_ids = [], []
            for cond, p_delta, p_scale, latent_mask in zip(
                conditions, position_deltas, position_scales, latent_masks
            ):
                # Prepare conditions
                c_latents, c_ids = encode_images(self.flux_pipe, cond)
                # Scale the position (see OminiConrtol2)
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                # Add position delta (see OminiControl)
                c_ids[:, 1] += p_delta[0][0]
                c_ids[:, 2] += p_delta[0][1]
                if len(p_delta) > 1:
                    print("Warning: only the first position delta is used.")
                # Append to the list
                if latent_mask is not None:
                    c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
        # Disable the attention cross different condition branches
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
        # Disable the attention from condition branches to image branch and text branch
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False

        # Forward pass
        transformer_out = transformer_forward(
            self.transformer,
            image_features=[x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            img_ids=[img_ids, *(condition_ids)],
            txt_ids=[text_ids],
            # There are three timesteps for the three branches
            # (text, image, and the condition)
            timesteps=[t, t] + [torch.zeros_like(t)] * len(conditions),
            # Same as above
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            guidances=[guidance] * branch_n,
            # The LoRA adapter names of each branch
            adapters=self.adapter_names,
            return_dict=False,
            group_mask=group_mask,
        )
        pred = transformer_out[0]

        # Compute loss
        step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()

        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx)
        # Ensure we return a tensor-wrapped loss for Lightning
        if isinstance(loss, torch.Tensor):
            return {"val_loss": loss.detach()}
        else:
            return {"val_loss": torch.tensor(loss)}

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


class TrainingCallback(L.Callback):
    def __init__(
        self,
        run_name,
        training_config: dict = {},
        test_function=None,
        val_dataloader=None,
    ):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.eval_interval = training_config.get("eval_interval", self.save_interval)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.test_function = test_function
        self.val_dataloader = val_dataloader
        if self.val_dataloader is None:
            print("TrainingCallback: no val_dataloader provided; validation hooks will be skipped")
        # max number of validation batches to run (0 = run full val set)
        self.val_max_batches = training_config.get("val_max_batches", 16)
        if self.val_max_batches == 0:
            print("TrainingCallback: val_max_batches=0 -> full validation dataset will be used")
        else:
            print(f"TrainingCallback: validating up to {self.val_max_batches} batches per eval")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Run validation and log val_loss to WandB (run locally to avoid trainer re-entrancy)
        if self.val_dataloader is not None and self.total_steps % self.eval_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Running validation"
            )

            val_loss = self._run_validation(pl_module)
            if self.use_wandb and val_loss is not None:
                wandb.log({"val_loss": val_loss}, step=getattr(trainer, "global_step", self.total_steps))
            pl_module.train()

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0 and self.test_function:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            pl_module.eval()
            self.test_function(
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
            )
            pl_module.train()

    def _move_batch_to_device(self, batch, device):
        # Recursively move tensors in batch to device
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_batch_to_device(v, device) for v in batch]
        elif isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(v, device) for v in batch)
        else:
            return batch

    def _run_validation(self, pl_module):
        # Run a lightweight validation loop over val_dataloader and return average val_loss
        pl_module.eval()
        device = getattr(pl_module, "device", None)
        is_cuda = device is not None and "cuda" in str(device)
        total = 0.0
        count = 0
        import time
        start_time = time.time()
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                # optionally limit number of validation batches
                if self.val_max_batches and self.val_max_batches > 0 and i >= self.val_max_batches:
                    break
                # Sanitize long text fields to avoid tokenizer/CLIP verbose truncation prints
                try:
                    max_seq = pl_module.model_config.get("max_sequence_length", 512)
                except Exception:
                    max_seq = 512
                max_chars = max_seq * 4
                if isinstance(batch, dict) and "description" in batch:
                    desc = batch["description"]
                    if isinstance(desc, str):
                        if len(desc) > max_chars:
                            batch["description"] = desc[:max_chars]
                    elif isinstance(desc, (list, tuple)):
                        batch["description"] = [d[:max_chars] if isinstance(d, str) and len(d) > max_chars else d for d in desc]

                if device is not None:
                    batch = self._move_batch_to_device(batch, device)
                out = pl_module.validation_step(batch, i)
                # Handle different return types
                if out is None:
                    continue
                if isinstance(out, dict):
                    val_loss = out.get("val_loss") or out.get("loss")
                else:
                    val_loss = out
                if val_loss is None:
                    continue
                if isinstance(val_loss, torch.Tensor):
                    value = float(val_loss.detach().cpu().item())
                else:
                    try:
                        value = float(val_loss)
                    except Exception:
                        continue
                total += value
                count += 1
        # sync and report timing
        if is_cuda:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        elapsed = time.time() - start_time
        if count > 0:
            print(f"Validation: processed {count} batches in {elapsed:.2f}s ({elapsed/count:.3f}s per batch), avg val_loss={total/count:.6f}")
        else:
            print(f"Validation: processed 0 batches in {elapsed:.2f}s")
        if count == 0:
            return None
        return total / count


def train(dataset, trainable_model, config, test_function, val_dataset=None):
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    # config = get_config()

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataloader
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )
    val_loader = None
    if val_dataset is not None:
        val_workers = training_config.get("val_dataloader_workers", training_config["dataloader_workers"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.get("batch_size", 1),
            shuffle=False,
            num_workers=val_workers,
        )

    # Callbacks for testing and saving checkpoints
    if is_main_process:
        callbacks = [
            TrainingCallback(run_name, training_config, test_function, val_dataloader=val_loader)
        ]
    print(f"val_loader: {'present' if val_loader is not None else 'none'}")

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks if is_main_process else [],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    if val_loader is not None:
        trainer.fit(trainable_model, train_loader, val_dataloaders=[val_loader])
    else:
        trainer.fit(trainable_model, train_loader)
