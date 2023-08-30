import argparse

NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "scale", "concatscale",
          "mpnnpoint", "mpnnpointTx"]


def add_args(parser):
    # model architecture options
    parser.add_argument('--input_dim', type=int, default=3,
                        help='Number of input dimensions (3 for 3D point clouds)')
    parser.add_argument('--dims', type=str, default="256",
                        help='Latent dimension for point cnf')
    parser.add_argument("--num_blocks", type=int, default=1,
                        help='Number of stacked CNFs for point generation')
    parser.add_argument('--use_cond_sampling', action='store_true',
                        help='Whether to use conditional sampling in latent CNF.')
    parser.add_argument("--layer_type_point", type=str, default="concatsquash", choices=LAYERS)
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
    parser.add_argument("--nonlinearity", type=str, default="swish", choices=NONLINEARITIES)
    parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--pos_enc', type=eval, default=False, choices=[True, False])
    parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)
    parser.add_argument('--kinetic_energy', type=float, default=None,
                        help="int_t ||f||_2^2")
    parser.add_argument('--jacobian_norm2', type=float, default=None,
                        help="int_t ||df/dx||_F^2")
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='std of Gaussian')

    # training options
    parser.add_argument('--zdim', type=int, default=128,
                        help='Dimension of the shape code')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use', choices=['adam', 'adamW', 'sgd'])
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size (of datasets) for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training (default: 100)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for initializing training. ')
    parser.add_argument('--recon_weight', type=float, default=1.,
                        help='Weight for the reconstruction loss.')
    parser.add_argument('--scheduler', type=str, default='linear',
                        help='Type of learning rate schedule')
    parser.add_argument("--warmup_iters", type=float, default=1000,
                        help='number of warmp iterations for scheduler')
    parser.add_argument('--lr_decay', type=float, default=1.,
                        help='Learning rate decay rate')
    parser.add_argument('--exp_decay_freq', type=int, default=1,
                        help='Learning rate exponential decay frequency')
    parser.add_argument('--clip_grad', type=float, default=0,
                        help='For value greater than 0, gradients are clipped accordingly')
    parser.add_argument('--train_subset', action='store_true', help='whether to train over 10% of training data')

    # data options
    parser.add_argument('--dataset_type', type=str, default="QM9",
                        help="Dataset types.", choices=['QM9', 'COD', 'ISO', 'GEOM_QM9_new', 'GEOM_QM9_v2', 'GEOM_Drugs', 'GEOM_Drugs_v2'])
    parser.add_argument('--new_features', action='store_true',
                        help='Whether to use new embedding node & edge features')
    parser.add_argument('--data_dir', type=str, default="data/",
                        help="Path to the training data")
    parser.add_argument('--node_dim', type=int, default=0,
                        help='Number of node feature dimensions')
    parser.add_argument('--edge_dim', type=int, default=0,
                        help='Number of edge feature dimensions')
    parser.add_argument('--global_dim', type=int, default=0,
                        help='Number of global node feature dimensions')
    parser.add_argument('--max_atoms', type=int, default=9,
                        help='Maximum number of atoms per molecule')
    parser.add_argument('--random_rotate', action='store_true',
                        help='Whether to randomly rotate each shape.')
    parser.add_argument('--normalize_per_shape', action='store_true',
                        help='Whether to perform normalization per shape.')
    parser.add_argument('--normalize_std_per_axis', action='store_true',
                        help='Whether to perform normalization per axis.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading threads')
    parser.add_argument('--virtual_node', action='store_true',
                        help='Whether to use virtual node')
    parser.add_argument('--standardise', action='store_true',
                        help='Whether to standardise all input features.')
    parser.add_argument('--cutoff_thres', type=float, default=1.,
                        help='cut off distance used in odefunc')

    # logging and saving frequency
    parser.add_argument('--log_name', type=str, default=None, help="Name for the log dir")
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=float, default=1.0)
    parser.add_argument('--viz_freq', type=int, default=10)

    # validation options
    parser.add_argument('--val_sampling', action='store_true',
                        help='Whether to evaluate sampling for validation or loss')
    parser.add_argument('--valsplit', default='val', type=str, choices=['val', 'train', 'test'],
                        help='data split to use for inference')

    # resuming
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to the checkpoint to be loaded.')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Whether to resume the optimizer when resumed training.')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    # Evaluation options
    parser.add_argument('--generate', action='store_true',
                        help='Generate samples')
    parser.add_argument('--val_sample_mol', default=1, type=int,
                        help='Number of samples per molecule')
    parser.add_argument('--mmff', action='store_true', help='Forcefield optimization')
    parser.add_argument('--onlyHeavy', action='store_true', help='only heavy atoms for evaluation')

    return parser


def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='Graph Conditional Continuous Normalizing Flow')
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
