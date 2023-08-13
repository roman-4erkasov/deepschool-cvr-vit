import os
import pandas as pd
import cv2
import torch as th
from typing import Optional
import lightning as lit
from PIL import Image
from config import Config
from dataset import Dataset
import torchvision as tv


class DataModule(lit.LightningDataModule):
    def __init__(
        self, 
        config: Config,
        transforms=None,
        debug=False,
    ):
        super().__init__()
        self.debug=debug
        self.num_workers=config.data_config.n_workers
        self.size = config.image_size
        self.data_dir = config.data_config.data_path
        self.ann_file = config.data_config.ann_file
        self.batch_size = config.data_config.batch_size
        if transforms is None:
            self.transform = tv.transforms.Compose(
                [
                    tv.transforms.Resize(
                        (self.size, self.size)
                    ),
                    tv.transforms.ToTensor(), 
                    # tv.transforms.Normalize(
                    #     (0.1307,), (0.3081,)
                    # ),
                ]
            )

    def setup(self, stage: str = None):
        if stage is None:
            stage = "fit"
        d_full = pd.read_csv(
            os.path.join(
                self.data_dir, self.ann_file
            )
        )
        if stage == "fit":
            d_train = d_full.loc[
                d_full["data set"] == "train",
                ["filepaths", "class id"]
            ]
            if self.debug:
                d_train = d_train.iloc[:5]
            self.train = Dataset(
                dir_path=self.data_dir, 
                annotation=d_train,
                transforms=self.transform
            )
            d_val = d_full.loc[
                d_full["data set"] == "valid",
                ["filepaths", "class id"]
            ]
            if self.debug:
                d_val = d_train.iloc[:5]
            self.val = Dataset(
                dir_path=self.data_dir, 
                annotation=d_val,
                transforms=self.transform
            )
            
        if stage == "test" or stage == "fit":
            d_test = d_full.loc[
                d_full["data set"] == "test",
                ["filepaths", "class id"]
            ]
            if self.debug:
                d_test = d_train.iloc[:5]
            self.test = Dataset(
                dir_path=self.data_dir, 
                annotation=d_test,
                transforms=self.transform
            )


    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(not self.debug)
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.val, 
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return th.utils.data.DataLoader(
            self.test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
