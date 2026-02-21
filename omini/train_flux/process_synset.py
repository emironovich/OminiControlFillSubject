import os
from datasets import load_dataset, Dataset

import sys
sys.path.append("/export/home/emironov/scripts/02_synset")

from synset2hugging_face import synset2hugging_face


def main():
    synset_dir = "/export/scratch/emironov/datasets/synset/SynsetSignsetGermany"
    image_list_csv = "CsvFiles/cyclesAll_train.csv"
    #image_list_csv = "CsvFiles/cyclesAll_val.csv"

    print("Processing synset dataset...")
    synset_ds = synset2hugging_face(synset_dir, image_list_csv)


    output_dir = "/export/scratch/emironov/datasets/synset_processed"
    os.makedirs(output_dir, exist_ok=True)

    synset_ds.save_to_disk(os.path.join(output_dir, "train"), num_shards=16)
    #synset_ds.save_to_disk(os.path.join(output_dir, "validation"), num_shards=16)

if __name__ == "__main__":
    main()
