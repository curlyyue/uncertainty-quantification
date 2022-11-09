import torch
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, 
                img_labels,       # annotation within desired distribution
                img_dir,           # full path of img directory. string
                transform=None, 
                target_transform=None):
        self.img_labels = img_labels
        self.img_dir = img_dir # directory
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        region = self.img_labels[idx][4]
        img_path = (f'{self.img_dir}/{self.img_labels[idx][1]}.npy')
        image = np.load(img_path)
        label = self.img_labels[idx][2]
        detail = self.img_labels[idx][3]
        flabel = [label, detail, region]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, flabel    


