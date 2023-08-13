import os
import pandas as pd
import cv2
import torch as th
from typing import Optional
from PIL import Image


class Dataset(th.utils.data.Dataset):
    def __init__(
        self, 
        dir_path: str, 
        annotation: pd.DataFrame,
        transforms=None
    ):
        self.dir_path=dir_path
        self.ann = annotation
        self.transforms=transforms
        
    def __len__(self):
        return self.ann.shape[0]

    def __getitem__(self, index: int):
        fpath = os.path.join(
            self.dir_path,
            self.ann["filepaths"].iloc[index]
        )
        image = Image.open(fpath)
        if self.transforms:
            image = self.transforms(image)
        class_id = self.ann["class id"].iloc[index]
        # print(f"[Dataset][__getitem__] {class_id.tolist()}")
        return image, class_id
