import pandas as pd
from dataset_manager.ClassificationDataset import MapillaryDataset
import os
import torch
import torchvision.transforms as transforms


def split_id_ood(config, split='train'):
    
    assert split in ['train', 'val', 'test']
    data_labels = pd.read_csv(config[f'{split}_csv'])
    id_data = data_labels[~data_labels.region.isin(config['ood_regions'])]
    ood_data = data_labels[data_labels.region.isin(config['ood_regions'])]
    
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])
    id_dataset = MapillaryDataset(id_data, 
                                os.path.join(config['dataset_path'], f'{split}_signs'),
                                transform=transform)
    ood_dataset = MapillaryDataset(ood_data, 
                                os.path.join(config['dataset_path'], f'{split}_signs'),
                                transform=transform)
    config['num_classes'] = id_data.label.nunique()

    id_dataloader = torch.utils.data.DataLoader(id_dataset, batch_size=config['batch_size'],
                                                 num_workers=6, pin_memory=True)

    ood_dataloader = torch.utils.data.DataLoader(ood_dataset, batch_size=config['batch_size'],
                                                 num_workers=6, pin_memory=True)

    return id_dataloader, ood_dataloader