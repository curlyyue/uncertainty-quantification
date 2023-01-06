import pandas as pd
from dataset_manager.ClassificationDataset import MapillaryDataset
import torch
import torchvision.transforms as transforms


def split_id_ood(config):
    
    train_data = pd.read_csv(config[f'train_csv'])
    val_data = pd.read_csv(config[f'val_csv'])
    test_data = pd.read_csv(config[f'test_csv'])
    ood_regions_classes = set(config['ood_regions'].split(','))
    regions = set(['g1', 'g2', 'g3', 'g4', 'g5', 'g6'])
    classes = set(['regu', 'warn', 'comp', 'info'])
    ood_regions = ood_regions_classes.intersection(regions)
    ood_classes = ood_regions_classes.intersection(classes)
    if ood_regions and ood_classes:
        raise ValueError('OOD should be either region OR class')
    unknown_ood = ood_regions_classes.difference(regions).difference(classes)
    if unknown_ood:
        raise ValueError('Check OOD config, you have provided unknown region or class', unknown_ood)
    
    train_id = train_data[~((train_data.region.isin(ood_regions)) | (train_data.label.isin(ood_classes)))]
    train_ood = train_data[((train_data.region.isin(ood_regions)) | (train_data.label.isin(ood_classes)))]
    val_id = val_data[~(((val_data.region.isin(ood_regions)) | (val_data.label.isin(ood_classes))))]
    val_ood = val_data[((val_data.region.isin(ood_regions)) | (val_data.label.isin(ood_classes)))]
    test_id = test_data[~(((test_data.region.isin(ood_regions)) | (test_data.label.isin(ood_classes))))]
    test_ood = test_data[((test_data.region.isin(ood_regions)) | (test_data.label.isin(ood_classes)))]
    ood = pd.concat([train_ood, val_ood, test_ood ])
    config['num_classes'] = train_id.label.nunique()
    print("Split done")
    print('Train ID', train_id.shape[0])
    print('Train OOD', train_ood.shape[0])
    print('Val ID', val_id.shape[0])
    print('Val OOD', val_ood.shape[0])
    print('Test ID', test_id.shape[0])
    print('Test OOD', test_ood.shape[0])
    print("Combined OOD", ood.shape[0])

    # data augmentation
    if config['augmentation'] == 'None':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif config['augmentation'] == 'AugMix':
        transform = transforms.Compose([
            transforms.AugMix(config['params']['AugMix']['severity'], config['params']['AugMix']['mixture_width']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif config['augmentation'] == 'AutoAug':
        policy = config['params']['AutoAug']['policy']
        if policy == 'ImageNet':
                policy = transforms.AutoAugmentPolicy.IMAGENET
        elif policy == 'CIFAR10':
                policy = transforms.AutoAugmentPolicy.CIFAR10
        elif policy == 'SVHN':
                policy = transforms.AutoAugmentPolicy.SVHN
        else:
                ValueError('Check policy, you have provided unknown region or class', policy)
        transform = transforms.Compose([
            transforms.AutoAugment(policy=policy),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif config['augmentation'] == 'TrivialAug':
        transform = transforms.Compose([
            transforms.TrivialAugmentWide(config['params']['TrivialAug']['num_mag_bins']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif config['augmentation'] == 'RandAug':
        param = tuple(config['params']['RandAug'].values())
        transform = transforms.Compose([
            transforms.RandAugment(param[0],param[1], param[2]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif config['augmentation'] == 'ManAug':
        param = tuple(config['params']['RandAug'].values())
        transform = transforms.Compose([
            transforms.AugMix(severity=1, mixture_width=1),
            transforms.RandAugment(num_ops=2,magnitude=3, num_magnitude_bins=31),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    else:
        raise ValueError('Check augmentation, you have provided unknown region or class', config['augmentation'])

    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])
    class_encoding = {c: i for i, c in enumerate(sorted(train_id.label.unique()))}
    train_id.loc[:, 'label_encoded'] = train_id.label.map(class_encoding)
    val_id.loc[:, 'label_encoded'] = val_id.label.map(class_encoding)
    test_id.loc[:, 'label_encoded'] = test_id.label.map(class_encoding)
    config['class_encoding'] = class_encoding
    config['num_classes'] = train_id.label.nunique()

    train_id_dataset = MapillaryDataset(train_id, transform=transform)
    val_id_dataset = MapillaryDataset(val_id, transform=transform_val_test)
    test_id_dataset = MapillaryDataset(test_id, transform=transform_val_test)
    test_ood_dataset = MapillaryDataset(test_ood, transform=transform_val_test)
    ood_dataset = MapillaryDataset(ood, transform=transform_val_test)

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


class Prepare_logits(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, x): 
        if self.source == 'huggingface':
            return torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))(x.last_hidden_state)

        return x