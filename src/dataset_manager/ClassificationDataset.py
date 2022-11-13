import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision


class MapillaryDataset(Dataset):
    def __init__(self, 
                img_labels,
                img_dir,
                transform=None, 
                target_transform=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        N = img_labels.label_encoded.value_counts().sort_index().values
        self.N = torch.tensor(N)
        self.output_dim = img_labels.label_encoded.nunique()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = f'{self.img_dir}/{self.img_labels.iloc[idx, 0]}.npy'
        image = np.load(img_path)
        if self.transform:
            image = self.transform(image)
        # image = image.permute(2, 0, 1)
        image = torchvision.transforms.functional.resize(image, [32, 32])
        label = self.img_labels.iloc[idx, -1]
 
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, torch.tensor([label])

