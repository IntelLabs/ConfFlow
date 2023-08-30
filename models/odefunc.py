import copy
import pickle
import torch
import torch.nn as nn
import numpy as np
from . import diffeq_layers


__all__ = ["ODEnet", "ODEfunc"]


def divergence_approx(f, y, e=None):

    samples = []
    sqnorms = []
    for e_ in e:
        e_dzdx = torch.autograd.grad(f, y, e_, create_graph=True)[0]
        n = e_dzdx.pow(2).sum(dim=-1, keepdim=True)
        sqnorms.append(n)
        e_dzdx_e = e_dzdx * e_
        samples.append(e_dzdx_e.sum(dim=-1, keepdim=True))

    S = torch.cat(samples, dim=-1)
    approx_tr_dzdx = S.mean(dim=-1, keepdim=True)

    N = torch.cat(sqnorms, dim=-1).mean(dim=-1, keepdim=True)

    return approx_tr_dzdx, N


class shifted_softplus(nn.Module):
    def __init__(self):
        super(shifted_softplus, self).__init__()
        self.act = nn.Softplus()
        self.shift = np.log(2.0)

    def forward(self, x):
        return self.act(x) - self.shift


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "ssoftplus": shifted_softplus(),
    "elu": nn.ELU(),
    "swish": nn.SiLU(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, hidden_dims, input_shape, context_dim, T_dim=1, layer_type="concat", nonlinearity="softplus",
                 node_dim=None, edge_dim=None, glbl_dim=None, cutoff=None):
        super(ODEnet, self).__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "squash": diffeq_layers.SquashLinear,
            "scale": diffeq_layers.ScaleLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "concatscale": diffeq_layers.ConcatScaleLinear,
            "mpnnpoint": diffeq_layers.MPNNPoint,
            "mpnnpointTx": diffeq_layers.PointTxFormer,
        }[layer_type]

        # build models and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape
        self.layer_kwargs = {'edge_dim': edge_dim, 'node_dim': node_dim, 'glbl_dim': glbl_dim, 'c': cutoff,
                        'act': NONLINEARITIES[nonlinearity]} if "mpnn" in layer_type else {}

        for dim_out in (hidden_dims + (input_shape[0],)):
            layer = base_layer(hidden_shape[0], dim_out, context_dim, T_dim, **self.layer_kwargs)
            layers.append(layer)
            activation_fns.append(copy.deepcopy(NONLINEARITIES[nonlinearity]))

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, context, y):
        dx = y
        for l, layer in enumerate(self.layers):
            context, dx = layer(context, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


class ODEfunc(nn.Module):
    def __init__(self, diffeq, point_conditional=False, L=4, rademacher=False, div_samples=1, label=None):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))
        self.pos_enc = torch.Tensor([2.0]).pow(torch.arange(0, L)).view(1, -1) * np.pi / 4
        self.point_conditional = point_conditional
        self.rademacher = rademacher
        self.div_samples = div_samples
        self.name = label

    def before_odeint(self, context, e=None):
        self.context = context
        self._e = e
        self._inter_x = []
        self._num_evals.fill_(0)

    def after_odeint(self):
        pickle.dump(self._inter_x, open(self.name + '.pkl', 'wb'))

    def forward(self, t, states):
        y = states[0]
        self._inter_x.append(y.detach().unsqueeze(0))
        if self.point_conditional:
            t = torch.ones(y.size(0), y.size(1), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
            t = t * self.pos_enc.unsqueeze(1).to(y)
        else:
            t = torch.ones(1, 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
            t = t * self.pos_enc.to(y)
        t = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)

        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = [torch.randint(low=0, high=2, size=y.shape).to(y).requires_grad_() * 2 - 1.
                           for _ in range(self.div_samples)]
            else:
                self._e = [torch.randn_like(y, requires_grad=True).to(y) for _ in range(self.div_samples)]

        with torch.set_grad_enabled(True):
            t.requires_grad_()

            if len(self.context) == 3 and len(states) == 6:  # conditional Graph CNF
                c = states[2]
                dy = self.diffeq(states[3:] + self.context + (t, c.view(*t.size()[:-1], -1)), y)
            elif len(self.context) == 3 and len(states) == 5:  # unconditional Graph CNF
                dy = self.diffeq(states[2:] + self.context + (t,), y)
            elif len(states) == 3:  # conditional CNF
                c = states[2]
                tc = torch.cat([t, c.view(*t.size()[:-1], -1)], dim=-1)
                dy = self.diffeq(tc, y)
            elif len(states) == 2:  # unconditional CNF
                dy = self.diffeq(t, y)
            else:
                assert 0, "`len(states)` should be 2 or 3"

            divergence, sqjacnorm = self.divergence_fn(dy, y, e=self._e)
            self.sqjacnorm = sqjacnorm

        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class RegularizedODEfunc(nn.Module):
    def __init__(self, odefunc, regularization_fns, conditional):
        super(RegularizedODEfunc, self).__init__()
        self.odefunc = odefunc
        self.regularization_fns = regularization_fns
        self.len_orig_state = 3 if conditional else 2
        if len(self.odefunc.diffeq.layer_kwargs) > 0:
            self.len_orig_state += 3

    def before_odeint(self, *args, **kwargs):
        self.odefunc.before_odeint(*args, **kwargs)

    def after_odeint(self):
        self.odefunc.after_odeint()

    def forward(self, t, state):

        with torch.enable_grad():
            t.requires_grad_(True)
            for s_ in state[:self.len_orig_state]:
                s_.requires_grad_()
            dstate = self.odefunc(t, state[:self.len_orig_state])
            if len(state) > self.len_orig_state:
                dx, dlogp = dstate[:2]
                reg_states = tuple(reg_fn(t, dx, dlogp, self.odefunc) for reg_fn in self.regularization_fns)
                return dstate + reg_states
            else:
                return dstate

    @property
    def _num_evals(self):
        return self.odefunc._num_evals


def quadratic_cost(t, dx, dlogp, context):
    del dlogp, t, context
    return 0.5*dx.pow(2).sum(dim=-1, keepdim=True)


def jacobian_frobenius_regularization_fn(t, dx, dlogp, context):
    del dlogp, t, dx
    return context.sqjacnorm
