import pandas as pd
from dataset_manager.ClassificationDataset import MapillaryDataset
import torch
import torchvision.transforms as transforms


def split_id_ood(config):
    
    train_data = pd.read_csv(config[f'train_csv'])
    val_data = pd.read_csv(config[f'val_csv'])
    test_data = pd.read_csv(config[f'test_csv'])
    ood_regions = config['ood_regions'].split(',')
    train_id = train_data[~train_data.region.isin(ood_regions)]
    train_ood = train_data[train_data.region.isin(ood_regions)]
    val_id = val_data[~val_data.region.isin(ood_regions)]
    val_ood = val_data[val_data.region.isin(ood_regions)]
    test_id = test_data[~test_data.region.isin(ood_regions)]
    test_ood = test_data[test_data.region.isin(ood_regions)]
    ood = pd.concat([train_ood, val_ood, test_ood ])
    config['num_classes'] = train_id.label.nunique()

    transform = transforms.Compose([transforms.ToTensor(),
        # data augmentation
        # transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1,2)),
        # transforms.ColorJitter(brightness=0.5, hue=0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])


    # if split == 'train':
    #     id_dataset = MapillaryDataset(id_data, 
    #                             os.path.join(config['dataset_path'], f'{split}_signs'),
    #                             transform=transform)
    #     ood_dataset = MapillaryDataset(ood_data, 
    #                             os.path.join(config['dataset_path'], f'{split}_signs'),
    #                             transform=transform)
    # else:
    #     id_dataset = MapillaryDataset(id_data, 
    #                             os.path.join(config['dataset_path'], f'{split}_signs'),
    #                             transform=None)
    #     ood_dataset = MapillaryDataset(ood_data, 
    #                             os.path.join(config['dataset_path'], f'{split}_signs'),
    #                             transform=None)

    config['num_classes'] = id_data.label.nunique()

    train_id_dataset = MapillaryDataset(train_id, transform=transform)
    val_id_dataset = MapillaryDataset(val_id, transform=transform)
    test_id_dataset = MapillaryDataset(test_id, transform=transform)
    test_ood_dataset = MapillaryDataset(test_ood, transform=transform)


    ood_dataset = MapillaryDataset(ood, transform=transform)

    train_id_dataloader = torch.utils.data.DataLoader(train_id_dataset,
                                                      batch_size=config['batch_size'],
                                                      num_workers=6, pin_memory=True)
    val_id_dataloader = torch.utils.data.DataLoader(val_id_dataset,
                                                      batch_size=config['batch_size'],
                                                      num_workers=6, pin_memory=True)
    test_id_dataloader = torch.utils.data.DataLoader(test_id_dataset,
                                                      batch_size=config['batch_size'],
                                                      num_workers=6, pin_memory=True)
    test_ood_dataloader = torch.utils.data.DataLoader(test_ood_dataset,
                                                      batch_size=config['batch_size'],
                                                      num_workers=6, pin_memory=True)
    ood_dataloader = torch.utils.data.DataLoader(ood_dataset, 
                                                batch_size=config['batch_size'],
                                                num_workers=6, pin_memory=True)

    return train_id_dataloader, val_id_dataloader, test_id_dataloader, test_ood_dataloader, ood_dataloader