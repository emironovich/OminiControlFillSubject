import os
from datasets import load_dataset, Dataset, Sequence, Value


def separate_images(ds, padding, image_size):
    new_images = []
    subjects = []
    new_descriptions = []
    for image, description in zip(ds["image"], ds["description"]):
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

        description_0 = description.get("description_0", "")
        new_descriptions.append(description_0)
    ds["image"] = new_images
    ds["subject"] = subjects
    ds["description"] = new_descriptions

    return ds



def process_Subjects200K(dataset, padding, image_size):
    """
    Prepairs the dataset to join with the SynSet dataset for training

    :param dataset: Dataset class
    :param padding: Padding size (from config)
    :param image_size: Image size (from config)
    :return: Dataset (HF) class with features "image", "description", "subject", "mask_bbox"
    """

    dataset = dataset.select_columns(["image", "description"])
    dataset = dataset.map(lambda x: separate_images(x, padding, image_size), batched=True, batch_size=100, num_proc=16)

    masks_list = [[] for _ in range(len(dataset))]

    masks_dataset = Dataset.from_dict({"mask_bbox": masks_list})
    dataset = dataset.add_column("mask_bbox", masks_dataset["mask_bbox"])
    dataset = dataset.cast_column("mask_bbox", Sequence(Value('int64')))

    return dataset






def main():
    # Initialize raw dataset
    raw_dataset = load_dataset("/export/scratch/emironov/datasets/Subjects200K")
    ds_train = raw_dataset["train"]
    split = ds_train.train_test_split(test_size=0.02, seed=42)
    raw_dataset = {"train": split["train"], "validation": split["test"]}

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

    padding = 8
    image_size = 512

    data_valid_val = raw_dataset["validation"].filter(
        filter_func,
        num_proc=16,
        cache_file_name="/export/scratch/emironov/cache/dataset/data_valid_val.arrow",
    )
    subject_ds_val = process_Subjects200K(data_valid_val, padding, image_size)

    output_dir = "/export/scratch/emironov/datasets/Subjects200K_processed"
    os.makedirs(output_dir, exist_ok=True)

    subject_ds_val.save_to_disk(os.path.join(output_dir, "validation"), num_shards=16)


    data_valid = raw_dataset["train"].filter(
        filter_func,
        num_proc=16,
        cache_file_name="/export/scratch/emironov/cache/dataset/data_valid_train.arrow",
    )
    subject_ds = process_Subjects200K(data_valid, padding, image_size)

    subject_ds.save_to_disk(os.path.join(output_dir, "train"), num_shards=16)

if __name__ == "__main__":
    main()
