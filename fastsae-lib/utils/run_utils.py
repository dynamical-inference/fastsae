import os

from datasets import load_dataset
from fastsae.dataset.clip_hf_dataset import HFDataset
from fastsae.utils.variables import DATASET_NAME_HF


def load_data(dataset, processor=None, split="train", shuffle=False, shuffle_seed=1):
    if isinstance(dataset, str):
        dataset_name_hf = DATASET_NAME_HF[dataset]
        dataset = load_dataset(dataset_name_hf, cache_dir=os.getenv("HF_HOME"), split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)
    clip_hf_dataset = HFDataset(dataset, processor) if processor is not None else dataset

    return clip_hf_dataset

def get_len_dataset(dataset):
    """Get the length of the dataset."""
    if hasattr(dataset, "num_rows"):
        if dataset.num_rows > 0:
            return dataset.num_rows
    else:
        return len(dataset["image"])
