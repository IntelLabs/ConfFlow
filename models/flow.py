import six
import ast
from .odefunc import ODEfunc, ODEnet, quadratic_cost, jacobian_frobenius_regularization_fn
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow
from .misc import PositionalEnc


REGULARIZATION_FNS = {
    "kinetic_energy": quadratic_cost,
    "jacobian_norm2": jacobian_frobenius_regularization_fn,
}


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(ast.literal_eval("args." + arg_key))

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


def build_model(args, input_dim, hidden_dims, context_dim, num_blocks,
                conditional, point_conditional, layer_type, nonlinearity,
                pos_enc=False, reg_fn=None):
    def build_cnf(label):
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(input_dim,),
            context_dim=context_dim,
            layer_type=layer_type + str(int(args.masking)),
            nonlinearity=nonlinearity,
            T_dim=8,
            node_dim=args.node_dim,
            edge_dim=args.edge_dim - 20, # subtracting 20 as edge feature contains distance proximity in the form of gaussian kernel
            glbl_dim=args.global_dim,
            cutoff=args.cutoff_thres,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
            point_conditional=point_conditional,
            L=4,
            rademacher=args.rademacher,
            label=str(label),
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            conditional=conditional,
            solver=args.solver,
            use_adjoint=args.use_adjoint,
            atol=args.atol,
            rtol=args.rtol,
            regularization_fns=reg_fn,
        )
        return cnf

    chain = [build_cnf(i) for i in range(num_blocks)]
    if pos_enc:
        pos_layers = [PositionalEnc(input_dim, args.max_atoms, mode='sin') for _ in range(num_blocks)]
    if args.batch_norm:
        bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag, sync=args.sync_bn)
                     for _ in range(num_blocks)]
        bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag, sync=args.sync_bn)]
        if pos_enc:
            for a, b, c in zip(pos_layers, chain, bn_layers):
                bn_chain.append(a)
                bn_chain.append(b)
                bn_chain.append(c)
        else:
            for a, b in zip(chain, bn_layers):
                bn_chain.append(a)
                bn_chain.append(b)
        chain = bn_chain
    model = SequentialFlow(chain)

    return model


def get_point_cnf(args, reg_fn, conditional=True):
    dims = tuple(map(int, args.dims.split("-")))
    model = build_model(args, args.input_dim, dims, args.zdim, args.num_blocks, conditional, args.use_cond_sampling,
                        args.layer_type_point, args.nonlinearity, args.pos_enc, reg_fn)
    print("Number of trainable parameters of Point CNF: {}".format(count_parameters(model)))
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)