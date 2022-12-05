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
# resize problem in dataset


def run(config): 

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

    wandb.init(project=config["wb_project"], entity="uncertainty_tum", 
                config=config, 
                id=config['save_dir'].split('/')[-1]
                )

    N = train_dataloader.dataset.N
    config['N'] = N.tolist()
    print("Datasets and Dataloaders created")

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
                             dropout=config['dropout'],
                             regr=config['regr'],
                             seed=config['seed'])

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

    ood_dataset_loaders = {}
    ood_dataset_loaders['Test OOD'] = ood_test_dataloader
    ood_dataset_loaders['Combined OOD'] = ood_dataloader
    result_path = os.path.join(config['save_dir'], 'best_model.pth')
    model.load_state_dict(torch.load(f'{result_path}')['model_state_dict'])
    metrics = test(model, test_dataloader, ood_dataset_loaders)
    
    config['metrics'] = metrics
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f)

    for k,v in metrics.items():
        if k != 'TestID_per_class_scores':
            print(k, round(v, 3))

    wandb.finish(quiet=True)

    return {**metrics}
