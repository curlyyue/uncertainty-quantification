import torch
import numpy as np
import os
import wandb


def compute_loss_accuracy(model, loader, device):
    model.eval()
    with torch.no_grad():
        loss = 0
        for batch_index, (X, Y) in enumerate(loader):
            Y_hot = torch.zeros(Y.shape[0], loader.dataset.output_dim)
            Y_hot.scatter_(1, Y, 1)
            X, Y_hot = X.to(device), Y_hot.to(device)
            Y_pred = model(X, Y_hot)
            if batch_index == 0:
                Y_pred_all = Y_pred.view(-1).to("cpu")
                Y_all = Y.view(-1).to("cpu")
            else:
                Y_pred_all = torch.cat([Y_pred_all, Y_pred.view(-1).to("cpu")], dim=0)
                Y_all = torch.cat([Y_all, Y.view(-1).to("cpu")], dim=0)
            loss += model.grad_loss.item()
        loss = loss / Y_pred_all.size(0)
        accuracy = ((Y_pred_all == Y_all).float().sum() / Y_pred_all.size(0)).item()
    model.train()
    return loss, accuracy


# Joint training for full model
def train(model, train_loader, val_loader, config):
    device = config['device']
    model.train()
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float("Inf")
    print('Training starts')
    for epoch in range(config['max_epochs']):
        for batch_index, (X_train, Y_train) in enumerate(train_loader):
            Y_train_hot = torch.zeros(Y_train.shape[0], train_loader.dataset.output_dim)
            Y_train_hot.scatter_(1, Y_train, 1)
            X_train, Y_train_hot = X_train.to(device), Y_train_hot.to(device)
            model(X_train, Y_train_hot)
            model.step()
            wandb.log({"Train loss": model.loss})

        # Stats on data sets
        train_loss, train_accuracy = compute_loss_accuracy(model, train_loader, device)
        train_losses.append(round(train_loss, 3))
        train_accuracies.append(round(train_accuracy, 3))

        val_loss, val_accuracy = compute_loss_accuracy(model, val_loader, config['device'])
        wandb.log({"Val loss": val_loss})
        wandb.log({"Val accuracy": val_accuracy})

        # val_losses.append(val_loss)
        # val_accuracies.append(val_accuracy)

        print("Epoch {} -> Val loss {} | Val Acc.: {}".format(epoch, round(val_loss, 3), round(val_accuracy, 3)))

        if np.isnan(val_loss):
            print('Detected NaN Loss')
            break

        if val_loss < -1.:
            print("Unstable training")
            break

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': val_accuracy}, 
                        os.path.join(config['save_dir'], 'best_model.pth'))
            print('Model saved')
        
        # fix early stopping
        # if epoch > patience and val_losses[-patience] <= min(val_losses[-patience:]):
        #     print('Early Stopping.')
        #     break

    return train_losses, val_losses, train_accuracies, val_accuracies


# Joint training method for ablated model
def train_sequential(model, train_loader, val_loader, max_epochs=200, frequency=2, patience=5, model_path='saved_model', full_config_dict={}):
    loss_1 = 'CE'
    loss_2 = model.loss

    print("### Encoder training ###")
    model.loss = loss_1
    model.no_density = True
    train_losses_1, val_losses_1, train_accuracies_1, val_accuracies_1 = train(model,
                                                                               train_loader,
                                                                               val_loader,
                                                                               max_epochs=max_epochs,
                                                                               frequency=frequency,
                                                                               patience=patience,
                                                                               model_path=model_path,
                                                                               full_config_dict=full_config_dict)
    print("### Normalizing Flow training ###")
    model.load_state_dict(torch.load(f'{model_path}')['model_state_dict'])
    for param in model.sequential.parameters():
        param.requires_grad = False
    model.loss = loss_2
    model.no_density = False
    train_losses_2, val_losses_2, train_accuracies_2, val_accuracies_2 = train(model,
                                                                               train_loader,
                                                                               val_loader,
                                                                               max_epochs=max_epochs,
                                                                               frequency=frequency,
                                                                               patience=patience,
                                                                               model_path=model_path,
                                                                               full_config_dict=full_config_dict)

    return train_losses_1 + train_losses_2, \
           val_losses_1 + val_losses_2, \
           train_accuracies_1 + train_accuracies_2, \
           val_accuracies_1 + val_accuracies_2
