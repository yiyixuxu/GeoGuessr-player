## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################

## Note : Editing in your Google Drive, changes will persist.
#############################################################



"""Base DataModule class and base Dataset class."""
from pathlib import Path
from typing import Collection, Any, Optional, Tuple, Union, Sequence, Callable
import os
from PIL import Image

import ml_collections

from torch.utils.data import DataLoader
import torch

import pytorch_lightning as pl


NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()

# sensible multiprocessing defaults: at most one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# but in distributed data parallel mode, we launch a training on each GPU, so must divide out to keep total at one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS
DEFAULT_GPUS = int(torch.cuda.is_available()) 
DEFAULT_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def pil_loader_default(path: Union[str, Path]) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def get_gps_default(img_name: Path):
    return {
      'latitude': float(img_name.stem.split('_')[1]),
      'longitude': float(img_name.stem.split('_')[2])
    }


class GPSBaseDataset(torch.utils.data.Dataset):
    """Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        img_paths:  Sequence,
        img_loader: Callable= pil_loader_default,
        get_gps_fn: Callable = get_gps_default,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        super().__init__()
        self.img_paths = img_paths
        self.img_loader = img_loader
        self.get_gps_fn = get_gps_fn
        self.transform = transform
        self.target_transform = target_transform
  

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """
        #datum, target = self.data[index], self.targets[index]
        path = self.img_paths[index]
        datum = self.img_loader(path)
        target = self.get_gps_fn(path)

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config:ml_collections.ConfigDict):
        super().__init__()
        self.data_dir = Path(config.data_dir)
        self.batch_size = config.batch_size
        self.num_workers = default(config.num_workers, DEFAULT_NUM_WORKERS)
        self.on_gpu = default(config.gpus,DEFAULT_GPUS)
        self.dataset_name = config.dataset
        self.extensions = default(config.extensions, DEFAULT_EXTENSIONS)

    def prepare_data(self, *args, **kwargs) -> None:
        """Take the first steps to prepare data for use.

        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """Perform final setup to prepare data for consumption by DataLoader.

        Here is where we typically split into train, validation, and test. This is done once per GPU in a DDP setting.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
    def train_dataloader(self):
        return DataLoader(
        self.data_train,
        shuffle=True,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=self.on_gpu,
    )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
