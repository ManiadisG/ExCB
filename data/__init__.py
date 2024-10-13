
from data.dataset_implementations import *
def build_dataset(data: str, data_set: str,dataset_path:str, args, logger, transforms=None):
    """
    - data: The name of the dataset
    - data_set: The set of the data to be used (train, val, etc)
    - dataset_path: The path to the dataset
    """
    if data.lower()=="imagenet":
        dataset = build_imagenet(dataset_path, data_set, args, transforms=transforms)
    else:
        raise ValueError("Unknown dataset")

    logger.print(f"Loaded {data} {data_set} set, samples no.: {len(dataset)}")
    return dataset