import torch
from src.results_manager.metrics_prior import confidence, brier_score, anomaly_detection
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
import wandb

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def compute_X_Y_alpha(model, loader, alpha_only=False):
    for batch_index, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        alpha_pred = model(X, None, return_output='alpha', compute_loss=False)
        if batch_index == 0:
            X_duplicate_all = X.to("cpu")
            orig_Y_all = Y.to("cpu")
            alpha_pred_all = alpha_pred.to("cpu")
        else:
            X_duplicate_all = torch.cat([X_duplicate_all, X.to("cpu")], dim=0)
            orig_Y_all = torch.cat([orig_Y_all, Y.to("cpu")], dim=0)
            alpha_pred_all = torch.cat([alpha_pred_all, alpha_pred.to("cpu")], dim=0)
    if alpha_only:
        return alpha_pred_all
    else:
        return orig_Y_all, X_duplicate_all, alpha_pred_all


def test(model, test_loader, ood_dataset_loaders):
    model.to(device)
    model.eval()
    class_names = test_loader.dataset.labels
    with torch.no_grad():
        orig_Y_all, X_duplicate_all, alpha_pred_all = compute_X_Y_alpha(model, test_loader)

        metrics = {}
        pred_classes = torch.max(alpha_pred_all, dim=-1)[1]
        metrics['accuracy_TestID'] = balanced_accuracy_score(orig_Y_all, pred_classes)
        metrics['TestID_per_class_scores'] = classification_report(orig_Y_all, pred_classes, 
                                             target_names=class_names, output_dict=True)
        print('TestID per class scores')
        print(classification_report(orig_Y_all, pred_classes, target_names=class_names))
        metrics['confidence_aleatoric'] = confidence(Y= orig_Y_all, alpha=alpha_pred_all,
                                                     score_type='APR', uncertainty_type='aleatoric')
        metrics['confidence_epistemic'] = confidence(Y= orig_Y_all, alpha=alpha_pred_all, 
                                                     score_type='APR', uncertainty_type='epistemic')
        metrics['brier_score'] = brier_score(Y= orig_Y_all, alpha=alpha_pred_all)

        wandb.log({'Test Accuracy': metrics['accuracy_TestID']})
        wandb.log({'Test ID aleatoric': metrics['confidence_aleatoric'], 
                   'Test ID epistemic': metrics['confidence_epistemic']})
        wandb.log({'Test ID brier': metrics['brier_score']})

        for ood_dataset_name, ood_loader in ood_dataset_loaders.items():
            ood_orig_Y_all, X_duplicate_all, ood_alpha_pred_all = compute_X_Y_alpha(model, ood_loader)
            pred_classes = torch.max(ood_alpha_pred_all, dim=-1)[1]

            metrics[f'accuracy_{ood_dataset_name}'] = balanced_accuracy_score(ood_orig_Y_all, pred_classes)
            metrics[f'anomaly_detection_aleatoric_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, 
                                                                                  ood_alpha=ood_alpha_pred_all, 
                                                                                  score_type='APR', 
                                                                                  uncertainty_type='aleatoric')
            metrics[f'anomaly_detection_epistemic_{ood_dataset_name}'] = anomaly_detection(alpha=alpha_pred_all, 
                                                                                  ood_alpha=ood_alpha_pred_all, 
                                                                                  score_type='APR', 
                                                                                  uncertainty_type='epistemic')
            wandb.log({f'{ood_dataset_name} aleatoric': metrics[f'anomaly_detection_aleatoric_{ood_dataset_name}'],
                       f'{ood_dataset_name} epistemic': metrics[f'anomaly_detection_epistemic_{ood_dataset_name}']})
    
    return metrics
