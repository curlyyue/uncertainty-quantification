config = dict(
seed = 1,  # Seed for training
dataset_path='/lab/project-1/',
train_csv='/lab/project-1/train_label.csv',
val_csv='/lab/project-1/test_label.csv', 
test_csv='/lab/project-1/test_label.csv',
# dataset_name,  # Dataset name. string
ood_regions=['g6'],  # OOD dataset regions (g1-g6).  list of strings
# unscaled_ood,  # If true consider also unscaled versions of ood datasets ?
# transform_min,  # Minimum value for rescaling input data. float ?
# transform_max,  # Maximum value for rescaling input data. float ?

# Architecture parameters
#model_save_dir = 'models/exp',  # Path to save model. string
architecture = 'resnet',  # Encoder architecture name. string
# input_dims,  # Input dimension. List of ints
# n_classes=4,  # Output dimension. int
hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
# kernel_dim,  # Input dimension. int
latent_dim=32,  # Latent dimension. int
no_density=False,  # Use density estimation or not. boolean
density_type='batched_radial_flow',  # Density type. string
n_density=6,  # Number of density components. int
k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
budget_function='id',  # Budget function name applied on class count. name

# Training parameters
save_dir = 'models/exp',  # Path to save resutls. string
max_epochs=200,  # Maximum number of epochs for training
patience=5,  # Patience for early stopping. int
# frequency,  # Frequency for early stopping test. int
batch_size=2,  # Batch size. int
lr=0.0001,  # Learning rate. float
loss='UCE',  # Loss name. string
training_mode='joint',  # 'joint' or 'sequential' training. string
regr=1e-5, # Regularization factor in Bayesian loss. float

WANDB_KEY = "ca13bbfeb8b55c13cc7a761af71fd11b88c907bf" 

)