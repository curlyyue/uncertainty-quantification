config = dict(
seed = 1,  # Seed for training
wb_project = 'test-project', #'clean_runs',
train_csv='/lab/project-1/train_label.csv',
val_csv='/lab/project-1/val_label.csv', 
test_csv='/lab/project-1/test_label.csv',
ood_regions='g6',  # OOD dataset regions (g1-g6). should be separated by , like g4,g6
# unscaled_ood,  # If true consider also unscaled versions of ood datasets ?
# transform_min,  # Minimum value for rescaling input data. float ?
# transform_max,  # Maximum value for rescaling input data. float ?

# Architecture parameters
architecture = 'resnet18',  # Encoder architecture name, should be from pretrained_info. string
hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
# kernel_dim,  # Input dimension. int
latent_dim=32,  # Latent dimension. int
no_density=False,  # Use density estimation or not. boolean
density_type='batched_radial_flow',  # Density type. string
n_density=6,  # Number of density components. int
k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
budget_function='id',  # Budget function name applied on class count. name

# Training parameters
save_dir = 'models/second_run',  # Path to save resutls. string
max_epochs=10,  # Maximum number of epochs for training 
patience=5,  # Patience for early stopping. int
batch_size=64,  # Batch size. int
lr=0.0001,  # Learning rate. float
loss='UCE',  # Loss name. string
training_mode='joint',  # 'joint' or 'sequential' training. string
regr=1e-5, # Regularization factor in Bayesian loss. float

# WANDB_KEY = "ca13bbfeb8b55c13cc7a761af71fd11b88c907bf" 

)

pretrained_info = dict(
    resnet18={'weights': "IMAGENET1K_V1", 'hidden_dim': 512},
    resnet50={'weights': "IMAGENET1K_V2", 'hidden_dim': 2048},
    efficientnetv2={'weights': "IMAGENET1K_V1", 'hidden_dim': 1280}
)