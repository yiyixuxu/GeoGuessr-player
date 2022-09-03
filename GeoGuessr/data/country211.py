
import pytorch_lightning as pl

from torchvision.datasets import Country211 as torch_Country211
from torchvision import transforms
import ml_collections

from typing import Callable 

from pathlib import Path
import os

from .base import BaseDataModule, GPSBaseDataset
from .util import get_files

class Country211(BaseDataModule):
    """The Country211 Data Set 
     <https://github.com/openai/CLIP/blob/main/data/country211.md>_ from OpenAI.

    This dataset was built by filtering the images from the YFCC100m dataset
    that have GPS coordinate corresponding to a ISO-3166 country code. The
    dataset is balanced by sampling 150 train images, 50 validation images, and
    100 test images images for each country.
    """

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__(config)
        self.transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
    def prepare_data(self):
        if not os.path.exists(self.data_dir / self.dataset_name):
            torch_Country211(self.data_dir, download = True)
    
    def setup(self, stage: str=None) -> None:
        if stage == 'fit' or stage is None:
            self.data_train = torch_Country211(self.data_dir, split = 'train', transform = self.transform)
            self.data_val = torch_Country211(self.data_dir, split = 'valid',transform = self.transform)
            self.class_to_idx = self.data_train.class_to_idx
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        if stage == "test" or stage is None:
            self.data_test = torch_Country211(self.data_dir, split='test',transform = self.transform)



class Country211_GPS(BaseDataModule):
    """The Country211 Data Set 
     <https://github.com/openai/CLIP/blob/main/data/country211.md>_ from OpenAI.

    This dataset was built by filtering the images from the YFCC100m dataset
    that have GPS coordinate corresponding to a ISO-3166 country code. The
    dataset is balanced by sampling 150 train images, 50 validation images, and
    100 test images images for each country.
    """

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__(config)
        self.transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
    

    def parepare_data(self):
        if not os.path.exists(self.data_dir / self.dataset_name):
            torch_Country211(self.data_dir, download = True)
 
    def setup(self, stage: str=None ) -> None:
  
        if stage == 'fit' or stage is None:
            img_paths_train = get_files(self.data_dir / self.dataset_name, extensions = self.extensions, folders=['train'])
            img_paths_val = get_files(self.data_dir / self.dataset_name, extensions = self.extensions, folders=['valid'])

            self.data_train = GPSBaseDataset(img_paths_train,transform = self.transform)
            self.data_val = GPSBaseDataset(img_paths_val,transform = self.transform)
        
        if stage == "test" or stage is None:
            img_paths_test = get_files(self.data_dir / self.dataset_name, extensions = self.extensions, folders=['test'])

            self.data_test = GPSBaseDataset(img_paths_test,transform = self.transform)

