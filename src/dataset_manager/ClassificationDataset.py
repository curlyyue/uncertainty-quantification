import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision


class MapillaryDataset(Dataset):
    def __init__(self, 
                data_df,
                transform=None):
        self.img_labels = data_df
        self.transform = transform
        N = self.img_labels.label_encoded.value_counts().sort_index().values
        self.N = torch.tensor(N)
        self.output_dim = self.img_labels.label_encoded.nunique()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx]['path']
        image = np.load(img_path)
        if self.transform:
            image = self.transform(image)
        image = torchvision.transforms.functional.resize(image, [64, 64])

        label = self.img_labels.iloc[idx]['label_encoded']

        return image, torch.tensor([label])

