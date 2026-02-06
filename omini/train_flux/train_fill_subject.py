import torch
import os
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset

from PIL import Image, ImageDraw

from datasets import load_dataset

from OminiControlFillSubject.omini.train_flux.train_subject import Subject200KDataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, generate


class FillSubject200KDataset(Subject200KDataset):
    def __getitem__(self, idx):
        # If target is 0, left image is target, right image is condition
        target = idx % 2
        item = self.base_dataset[idx // 2]

        # Crop the image to target and condition
        image = item["image"]
        left_img = image.crop(
            (
                self.padding,
                self.padding,
                self.image_size + self.padding,
                self.image_size + self.padding,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )

        # Get the target and condition image
        target_image, condition_img_subject = (
            (left_img, right_img) if target == 0 else (right_img, left_img)
        )

        # Resize the image
        condition_img_subject = condition_img_subject.resize(self.condition_size).convert("RGB")
        target_image = target_image.resize(self.target_size).convert("RGB")

        # Get the description
        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]


        condition_size = self.condition_size

        condition_imgs, position_deltas = [], []
        for c_type in self.condition_type:
            if c_type == "subject":
                condition_img = condition_img_subject
                # 16 is the downscale factor of the image.
                # More details about position delta can be found in the documentation.
                position_delta = np.array([0, -self.condition_size[0] // 16])
            elif c_type == "fill":
                condition_img= target_image.copy().resize(condition_size).convert("RGB")
                w, h = condition_img.size
                x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
                y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
                mask = Image.new("L", condition_img.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle([x1, y1, x2, y2], fill=255)
                if random.random() > 0.5:
                    mask = Image.eval(mask, lambda a: 255 - a)
                condition_img = Image.composite(
                    condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
                )

                position_delta = np.array([0, 0])
            else:
                raise ValueError(f"Condition type {c_type} is not  implemented.")

            condition_imgs.append(condition_img.convert("RGB"))
            position_deltas.append(position_delta)

        # Randomly drop text or image (for training)
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob

        if drop_text:
            description = ""
        if drop_image:
            condition_imgs = [
                Image.new("RGB", condition_size)
                for _ in range(len(self.condition_type))
            ]

        return_dict = {
            "image": self.to_tensor(image),
            "description": description,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        }

        # TODO can this position scale be different for different conditions?
        position_scale = 1.0
        for i, c_type in enumerate(self.condition_type):
            return_dict[f"condition_{i}"] = self.to_tensor(condition_imgs[i])
            return_dict[f"condition_type_{i}"] = self.condition_type[i]
            return_dict[f"position_delta_{i}"] = position_deltas[i]
            return_dict[f"position_scale_{i}"] = position_scale

        return return_dict



@torch.no_grad()
def test_function(model, save_path, file_name):
    # TODO: Implement the logic to generate a sample using the model
    raise NotImplementedError("Sample generation not implemented")


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Initialize custom dataset
    dataset = CustomDataset()

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
