import torch
import os
import sys
sys.path.append("../")
from src.posterior_networks.PosteriorNetwork import PosteriorNetwork
from src.posterior_networks.train import train, train_sequential
from src.posterior_networks.test import test
from src.posterior_networks.utils import split_id_ood
from config import config
import json
import wandb
import numpy as np

# to do
# augmentations?
# resize problem in dataset
# check N
# check k_lipschitz
# check unscaled_ood


def run(config): 

    # fix seeds
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    os.makedirs(config['save_dir'], exist_ok=True)

    use_cuda = torch.cuda.is_available()
    config['device'] = "cuda" if use_cuda else "cpu"

    ##################
    ## Load dataset ##
    ##################
    (train_dataloader, 
     val_dataloader, 
     test_dataloader, 
     ood_test_dataloader, 
     ood_dataloader
    ) = split_id_ood(config)
    N = train_dataloader.dataset.N
    print("Datasets and Dataloaders created")

    wandb.init(project=config["wb_project"], entity="uncertainty_tum", 
                  config=config)
    #################
    ## Train model ##
    #################
    model = PosteriorNetwork(N=N,
                             n_classes=config['num_classes'],
                             hidden_dims=config['hidden_dims'],
                             kernel_dim=None,
                             latent_dim=config['latent_dim'],
                             architecture=config['architecture'],
                             k_lipschitz=config['k_lipschitz'],
                             no_density=config['no_density'],
                             density_type=config['density_type'],
                             n_density=config['n_density'],
                             budget_function=config['budget_function'],
                             batch_size=config['batch_size'],
                             lr=config['lr'],
                             loss=config['loss'],
                             regr=config['regr'],
                             seed=config['seed'])

    # full_config_name = ''
    # for k, v in full_config_dict.items():
    #     full_config_name += str(v) + '-'
    # full_config_name = full_config_name[:-1]
    # model_path = f'{directory_model}/model-dpn-{full_config_name}'
    model.to(config['device'])
    if config['training_mode'] == 'joint':
        train(model, train_dataloader, val_dataloader, config=config)
    elif config['training_mode'] == 'sequential':
        assert not config['no_density']
        # should be fixed
        train_losses, val_losses, train_accuracies, val_accuracies = train_sequential(model,
                                                                                       train_loader,
                                                                                       val_loader,
                                                                                       max_epochs=max_epochs,
                                                                                       frequency=frequency,
                                                                                       patience=patience,
                                                                                       model_path=model_path,
                                                                                       full_config_dict=full_config_dict)
        pass
    else:
        raise NotImplementedError

    ################
    ## Test model ##
    ################

    # for ood_dataset_name in ood_dataset_names:
    #     if unscaled_ood:
            # dataset = ClassificationDataset(f'{directory_dataset}/{ood_dataset_name}.csv',
            #                                 input_dims=input_dims, output_dim=output_dim,
            #                                 seed=None)
            # ood_dataset_loaders[ood_dataset_name + '_unscaled'] = torch.utils.data.DataLoader(dataset, batch_size=2 * batch_size, num_workers=4, pin_memory=True)
    ood_dataset_loaders = {}
    ood_dataset_loaders['Test OOD'] = ood_test_dataloader
    ood_dataset_loaders['Combined OOD'] = ood_dataloader
    result_path = os.path.join(config['save_dir'], 'best_model.pth')
    model.load_state_dict(torch.load(f'{result_path}')['model_state_dict'])
    metrics = test(model, test_dataloader, ood_dataset_loaders, result_path)
    
    config['metrics'] = metrics
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f)

    for k,v in metrics.items():
        print(k, round(v, 3))

    wandb.finish()

    return {**metrics}
