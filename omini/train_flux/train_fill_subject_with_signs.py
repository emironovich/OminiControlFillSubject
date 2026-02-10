import torch
import os
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset

from PIL import Image, ImageDraw

from datasets import load_dataset, concatenate_datasets

from .train_subject import Subject200KDataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, generate
import sys
sys.path.append("/export/home/emironov/scripts/02_synset")

from synset2hugging_face import synset2hugging_face


class FillSubjectDataset(Dataset):
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        target_image = item["image"]
        subject_image = item["subject"]

        subject_image = subject_image.resize(self.condition_size).convert("RGB")
        target_image = target_image.resize(self.target_size).convert("RGB")

        # Get the description
        description = item["description"]

        condition_size = self.condition_size

        condition_imgs, position_deltas = [], []
        for c_type in self.condition_type:
            if c_type == "subject":
                condition_img = subject_image
                # 16 is the downscale factor of the image.
                # More details about position delta can be found in the documentation.
                position_delta = np.array([0, -self.condition_size[0] // 16])
            elif c_type == "fill":
                mask_bbox = item["mask_bbox"]
                flip_mask = False
                if mask_bbox and random.random() > 0.5:
                    x1, y1, w, h = mask_bbox
                    x2, y2 = x1 + w, y1 + h
                else:
                    ww, hh = condition_img.size
                    x1, x2 = sorted([random.randint(0, ww - 1), random.randint(0, ww - 1)])
                    y1, y2 = sorted([random.randint(0, hh - 1), random.randint(0, hh - 1)])
                    flip_mask = random.random() > 0.5
 
                condition_img = target_image.copy().resize(condition_size).convert("RGB")
                mask = Image.new("L", condition_img.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle([x1, y1, x2, y2], fill=255)
                if not flip_mask:
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
        if drop_image and len(description) > 0:
            condition_imgs = [
                Image.new("RGB", condition_size)
                for _ in range(len(self.condition_type))
            ]

        return_dict = {
            "image": self.to_tensor(target_image),
            "description": description,
            **({"pil_image": [target_image, condition_imgs]} if self.return_pil_image else {}),
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
    condition_size = model.training_config["dataset"]["condition_size"]
    target_size = model.training_config["dataset"]["target_size"]


    condition_type = model.training_config["condition_type"]
    test_list = []

    condition_list = []
    # Test case1 (in-distribution test case)
    prompt = "Resting on the picnic table at a lakeside campsite, it's caught in the golden glow of early morning, with mist rising from the water and tall pines casting long shadows behind the scene."
    for i, c_type in enumerate(condition_type):
        if c_type == "subject":
            image = Image.open("assets/test_in.jpg")
            image = image.resize(condition_size).convert("RGB")
            # More details about position delta can be found in the documentation.
            position_delta = np.array([0, -condition_size[0] // 16])

            condition = Condition(image, model.adapter_names[i + 2], position_delta)
        else:
            image = Image.open("assets/pexels_table.jpg")
            image = image.resize(condition_size)
            x1, x2 = 220, 300
            y1, y2 = 200, 280
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            ).convert("RGB")
            position_delta = np.array([0, 0])
            condition = Condition(condition_img, model.adapter_names[i + 2], position_delta)
        condition_list.append(condition)

    test_list.append((condition_list, prompt))

    condition_list = []
    prompt = "In a bright room. It is placed on a sofa."

    for i, c_type in enumerate(condition_type):
        if c_type == "subject":
            image = Image.open("assets/test_out.jpg")
            image = image.resize(condition_size)
            # More details about position delta can be found in the documentation.
            position_delta = np.array([0, -condition_size[0] // 16])

            condition = Condition(image.convert("RGB"), model.adapter_names[i + 2], position_delta)
        else:
            image = Image.open("assets/room_corner.jpg")
            image = image.resize(condition_size)
            x1, x2 = 120, 200
            y1, y2 = 220, 300
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            ).convert("RGB")
            position_delta = np.array([0, 0])
            condition = Condition(condition_img, model.adapter_names[i + 2], position_delta)
        condition_list.append(condition)

    test_list.append((condition_list, prompt))

    condition_list = []
    prompt = "It is outside with a landscape background."

    for i, c_type in enumerate(condition_type):
        if c_type == "subject":
            image = Image.open("assets/Stop.png")
            image = image.resize(condition_size)
            # More details about position delta can be found in the documentation.
            position_delta = np.array([0, -condition_size[0] // 16])

            condition = Condition(image.convert("RGB"), model.adapter_names[i + 2], position_delta)
        else:
            # this image already has a mask
            image = Image.open("assets/85_0000061.jpg")
            image = image.resize(condition_size)
            position_delta = np.array([0, 0])
            condition = Condition(image.convert("RGB"), model.adapter_names[i + 2], position_delta)
        condition_list.append(condition)

    test_list.append((condition_list, prompt))



    # Generate images
    os.makedirs(save_path, exist_ok=True)
    for i, (condition_list, prompt) in enumerate(test_list):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)

        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=condition_list,
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
        )
        file_path = os.path.join(save_path, f"{file_name}_fill_subject_{i}.jpg")
        res.images[0].save(file_path)

def process_Subjects200K(dataset, padding, image_size):
    """
    Prepairs the dataset to join with the SynSet dataset for training
    
    :param dataset: Dataset class
    :param padding: Padding size (from config)
    :param image_size: Image size (from config)
    :return: Dataset class with features "image", "description", "subject", "mask_bbox"
    """
    def separate_images(ds, padding, image_size):
        new_images = []
        subjects = []
        for image in ds["image"]:
            left_img = image.crop(
                (
                    padding,
                    padding,
                    image_size + padding,
                    image_size + padding,
                )
            )
            right_img = image.crop(
                (
                    image_size + padding * 2,
                    padding,
                    image_size * 2 + padding * 2,
                    image_size + padding,
                )
            )
            new_images.append(left_img)
            subjects.append(right_img)
        ds["image"] = new_images
        ds["subject"] = subjects

        return ds


    dataset = dataset.select_columns(["image", "description"])
    #dataset = dataset.map(lambda x: separate_images(x, padding, image_size), batched=True, num_proc=16)
    dataset = dataset.map(lambda x: separate_images(x, padding, image_size), batched=True, batch_size=100, num_proc=16)

    masks_list = [[] for _ in range(len(dataset))]

    masks_dataset = Dataset.from_dict({"mask_bbox": masks_list})
    dataset = dataset.add_column("mask_bbox", masks_dataset["mask_bbox"])

    return dataset



    


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Initialize raw dataset
    raw_dataset = load_dataset("/export/scratch/emironov/datasets/Subjects200K")

    # Define filter function to filter out low-quality images from Subjects200K
    def filter_func(item):
        if not item.get("quality_assessment"):
            return False
        return all(
            item["quality_assessment"].get(key, 0) >= 5
            for key in ["compositeStructure", "objectConsistency", "imageQuality"]
        )

    # Filter dataset
    if not os.path.exists("/export/scratch/emironov/cache/dataset"):
        os.makedirs("/export/scratch/emironov/cache/dataset")
    data_valid = raw_dataset["train"].filter(
        filter_func,
        num_proc=16,
        cache_file_name="/export/scratch/emironov/cache/dataset/data_valid.arrow",
    )

    subject_ds = process_Subjects200K(data_valid, training_config["dataset"]["padding"], training_config["dataset"]["image_size"]) 

    synset_dir = "/export/scratch/emironov/datasets/synset/SynsetSignsetGermany"
    image_list_csv = "CsvFiles/cyclesAll_train.csv"

    synset_ds = synset2hugging_face(synset_dir, image_list_csv)

    ds = concatenate_datasets([subject_ds, synset_ds])
   

    # Initialize the dataset
    dataset = FillSubjectDataset(
        ds,
        condition_size=training_config["dataset"]["condition_size"],
        target_size=training_config["dataset"]["target_size"],
        image_size=training_config["dataset"]["image_size"],
        padding=training_config["dataset"]["padding"],
        condition_type=training_config["condition_type"],
        drop_text_prob=training_config["dataset"]["drop_text_prob"],
        drop_image_prob=training_config["dataset"]["drop_image_prob"],
    )

    cond_n = len(training_config["condition_type"])

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=[None, None, *["default"] * cond_n],
        # In this setting, all the conditions are using the same LoRA adapter
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
