import torch
from torch import nn
from torch.distributions.dirichlet import Dirichlet
from transformers import AutoModelForImageClassification

from src.architectures.linear_sequential import linear_sequential
from src.posterior_networks.NormalizingFlowDensity import NormalizingFlowDensity
from src.posterior_networks.BatchedNormalizingFlowDensity import BatchedNormalizingFlowDensity
from src.posterior_networks.MixtureDensity import MixtureDensity
from src.posterior_networks.config import pretrained_info
from src.posterior_networks.utils import Prepare_logits

__budget_functions__ = {'one': lambda N: torch.ones_like(N),
                        'log': lambda N: torch.log(N + 1.),
                        'id': lambda N: N,
                        'id_normalized': lambda N: N / N.sum(),
                        'exp': lambda N: torch.exp(N),
                        'parametrized': lambda N: torch.nn.Parameter(torch.ones_like(N).to(N.device))}


class PosteriorNetwork(nn.Module):
    def __init__(self, N,  # Count of data from each class in training set. list of ints
                 n_classes,  # Output dimension. int
                 hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
                 kernel_dim=None,  # Kernel dimension if conv architecture. int
                 latent_dim=10,  # Latent dimension. int
                 architecture='linear',  # Encoder architecture name. int
                 k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                 no_density=False,  # Use density estimation or not. boolean
                 density_type='radial_flow',  # Density type. string
                 n_density=8,  # Number of density components. int
                 budget_function='id',  # Budget function name applied on class count. name
                 batch_size=64,  # Batch size. int
                 lr=1e-3,  # Learning rate. float
                 loss='UCE',  # Loss name. string
                 dropout=0.1,
                 regr=1e-5,  # Regularization factor in Bayesian loss. float
                 seed=123):  # Random seed for init. int
        super().__init__()

        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type(torch.DoubleTensor)

        # Architecture parameters
        self.output_dim, self.hidden_dims, self.kernel_dim, self.latent_dim = n_classes, hidden_dims, kernel_dim, latent_dim
        self.k_lipschitz = k_lipschitz
        self.no_density, self.density_type, self.n_density = no_density, density_type, n_density
        if budget_function in __budget_functions__:
            self.N, self.budget_function = __budget_functions__[budget_function](N), budget_function
        else:
            raise NotImplementedError
        # Training parameters
        self.batch_size, self.lr = batch_size, lr
        self.loss, self.regr = loss, regr

        # Encoder -- Feature selection
        if architecture not in pretrained_info:
            raise NotImplementedError
        model_info = pretrained_info[architecture]
        if model_info['source'] == 'torchvision':
            model = torch.hub.load("pytorch/vision", architecture, 
                                weights=model_info['weights'],
                                )
        elif model_info['source'] == 'huggingface':
            model = AutoModelForImageClassification.from_pretrained(model_info['weights'])
        
        self.sequential = torch.nn.Sequential(*(list(model.children())[:-1]),
                                    Prepare_logits(model_info['source']),
                                    nn.Flatten(),
                                    torch.nn.Dropout(p=dropout),
                                    nn.Linear(model_info['hidden_dim'], self.latent_dim)
        )
        self.batch_norm = nn.BatchNorm1d(num_features=self.latent_dim)
        self.linear_classifier = linear_sequential(input_dims=[self.latent_dim],  # Linear classifier for sequential training
                                                   hidden_dims=[self.hidden_dims[-1]],
                                                   output_dim=self.output_dim,
                                                   k_lipschitz=self.k_lipschitz)

        # Normalizing Flow -- Normalized density on latent space
        if self.density_type == 'planar_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type) for c in range(self.output_dim)])
        elif self.density_type == 'radial_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type) for c in range(self.output_dim)])
        elif self.density_type == 'batched_radial_flow':
            self.density_estimation = BatchedNormalizingFlowDensity(c=self.output_dim, dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type.replace('batched_', ''))
        elif self.density_type == 'iaf_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type) for c in range(self.output_dim)])
        elif self.density_type == 'normal_mixture':
            self.density_estimation = nn.ModuleList([MixtureDensity(dim=self.latent_dim, n_components=n_density, mixture_type=self.density_type) for c in range(self.output_dim)])
        else:
            raise NotImplementedError
        self.softmax = nn.Softmax(dim=-1)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, input, soft_output, return_output='hard', compute_loss=True):
        batch_size = input.size(0)
        if self.N.device != input.device:
            self.N = self.N.to(input.device)

        if self.budget_function == 'parametrized':
            N = self.N / self.N.sum()
        else:
            N = self.N

        # Forward
        zk = self.sequential(input)
        if self.no_density:  # Ablated model without density estimation
            logits = self.linear_classifier(zk)
            alpha = torch.exp(logits)
            soft_output_pred = self.softmax(logits)
        else:  # Full model with density estimation
            zk = self.batch_norm(zk)
            log_q_zk = torch.zeros((batch_size, self.output_dim)).to(zk.device.type)
            alpha = torch.zeros((batch_size, self.output_dim)).to(zk.device.type)

            if isinstance(self.density_estimation, nn.ModuleList):
                for c in range(self.output_dim):
                    log_p = self.density_estimation[c].log_prob(zk)
                    log_q_zk[:, c] = log_p
                    alpha[:, c] = 1. + (N[c] * torch.exp(log_q_zk[:, c]))
            else:
                log_q_zk = self.density_estimation.log_prob(zk)
                alpha = 1. + (N[:, None] * torch.exp(log_q_zk)).permute(1, 0)

            pass

            soft_output_pred = torch.nn.functional.normalize(alpha, p=1)
        output_pred = self.predict(soft_output_pred)

        # Loss
        if compute_loss:
            if self.loss == 'CE':
                self.grad_loss = self.CE_loss(soft_output_pred, soft_output)
            elif self.loss == 'UCE':
                self.grad_loss = self.UCE_loss(alpha, soft_output)
            else:
                raise NotImplementedError

        if return_output == 'hard':
            return output_pred
        elif return_output == 'soft':
            return soft_output_pred
        elif return_output == 'alpha':
            return alpha
        elif return_output == 'latent':
            return zk
        else:
            raise AssertionError

    def CE_loss(self, soft_output_pred, soft_output):
        # with autograd.detect_anomaly():
        CE_loss = - torch.sum(soft_output.squeeze() * torch.log(soft_output_pred))

        return CE_loss

    def UCE_loss(self, alpha, soft_output):
        # with autograd.detect_anomaly():
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.output_dim)
        entropy_reg = Dirichlet(alpha).entropy()
        UCE_loss = torch.sum(soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))) - self.regr * torch.sum(entropy_reg)
        return UCE_loss

    def step(self):
        self.optimizer.zero_grad()
        self.grad_loss.backward()
        self.optimizer.step()

    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred
