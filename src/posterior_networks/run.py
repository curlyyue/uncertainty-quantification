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

# to do
# fix seeds
# check N
# make test data
# logic for ood evaluation and test()
# resize problem in dataset
# early stopping
# inputs dims?
# put some prints

def run(config): 

    os.makedirs(config['save_dir'], exist_ok=True)
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f)
    
    wandb.init(project="test-project", entity="uncertainty_tum")
    wandb.config = config
    
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    config['device'] = device
    torch.backends.cudnn.benchmark = True

    ##################
    ## Load dataset ##
    ##################

    train_dataloader, ood_train_dataloader = split_id_ood(config, split='train')
    val_dataloader, ood_val_dataloader = split_id_ood(config, split='val')
    test_dataloader, ood_test_dataloader = split_id_ood(config, split='test')
    N = train_dataloader.dataset.N # fix this?
    print("Datasets and Dataloaders created")
    # dataset = ClassificationDataset(f'{directory_dataset}/{dataset_name}.csv',
    #                                 input_dims=input_dims, output_dim=output_dim,
    #                                 transform_min=transform_min, transform_max=transform_max,
    #                                 seed=seed_dataset)
    # train_loader, val_loader, test_loader, N = dataset.split(batch_size=batch_size, split=split, num_workers=4)

    #################
    ## Train model ##
    #################
    model = PosteriorNetwork(N=N,
                             #input_dims=input_dims,
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
    model.to(device)
    if config['training_mode'] == 'joint':
        train(model, train_dataloader, val_dataloader, config=config)
    elif config['training_mode'] == 'sequential':
        assert not config['no_density']
        # should be fixed
        # train_losses, val_losses, train_accuracies, val_accuracies = train_sequential(model,
        #                                                                                train_loader,
        #                                                                                val_loader,
        #                                                                                max_epochs=max_epochs,
        #                                                                                frequency=frequency,
        #                                                                                patience=patience,
        #                                                                                model_path=model_path,
        #                                                                                full_config_dict=full_config_dict)
        pass
    else:
        raise NotImplementedError

    ################
    ## Test model ##
    ################
    # ood_dataset_loaders = {}
    # for ood_dataset_name in ood_dataset_names:
        
        # if unscaled_ood:
        #     dataset = ClassificationDataset(f'{directory_dataset}/{ood_dataset_name}.csv',
        #                                     input_dims=input_dims, output_dim=output_dim,
        #                                     seed=None)
            # ood_dataset_loaders[ood_dataset_name + '_unscaled'] = torch.utils.data.DataLoader(dataset, batch_size=2 * batch_size, num_workers=4, pin_memory=True)
    # result_path = os.path.join(config['save_dir'], 'best_model.pth')
    # model.load_state_dict(torch.load(f'{result_path}')['model_state_dict'])
    # metrics = test(model, test_loader, ood_dataset_loaders, result_path)

    # results = {
    #     'train_losses': train_losses,
    #     'val_losses': val_losses,
    #     'train_accuracies': train_accuracies,
    #     'val_accuracies': val_accuracies,
    # }
    wandb.finish()


    # return {**results, **metrics}
