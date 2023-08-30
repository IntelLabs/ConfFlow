import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from .odefunc import RegularizedODEfunc


__all__ = ["CNF", "SequentialFlow"]


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, context, logpx=None, reg_states=tuple(), reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, context, logpx, reg_states, integration_times, reverse)
            return x
        else:
            for i in inds:
                x, logpx, reg_states = self.chain[i](x, context, logpx, reg_states, integration_times, reverse)
            return x, logpx, reg_states


class CNF(nn.Module):
    def __init__(self, odefunc, conditional=True, T=1.0, train_T=False, regularization_fns=None,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None and len(regularization_fns) > 0:
            nreg = len(regularization_fns)
            odefunc = RegularizedODEfunc(odefunc, regularization_fns, conditional)

        self.odefunc = odefunc
        self.use_adjoint = use_adjoint
        self.nreg = nreg
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.test_solver_options = {}
        self.conditional = conditional

    def forward(self, x, context=None, logpx=None, reg_states=tuple(), integration_times=None, reverse=False):

        if not len(reg_states) == self.nreg:
            reg_states = tuple(torch.zeros(*x.shape[:-1], 1).to(x) for _ in range(self.nreg))

        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
        else:
            _logpx = logpx

        addl_states = 0
        if self.conditional:
            assert context is not None
            states = (x, _logpx, context[0], context[1], context[2], context[3])
            context = context[4:]
            addl_states = 4
        else:
            if context is None:
                states = (x, _logpx)
                context = tuple()
            else:
                states = (x, _logpx, context[0], context[1], context[2])
                context = context[3:] + (torch.bincount(context[4]).tolist(), )
                addl_states = 3

        atol = [self.atol, self.atol] + [1e20] * (self.nreg + addl_states) if self.solver in ['dopri5', 'bosh3'] else self.atol
        rtol = [self.rtol, self.rtol] + [1e20] * (self.nreg + addl_states) if self.solver in ['dopri5', 'bosh3'] else self.rtol

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(x)

        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics and pass on context data
        self.odefunc.before_odeint(context=context)

        if self.training:
            state_t = odeint(
                self.odefunc,
                states + reg_states,
                integration_times.to(x),
                atol=atol,
                rtol=rtol,
                adjoint_rtol=self.rtol,
                adjoint_atol=self.atol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                states + reg_states,
                integration_times.to(x),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
                options=self.test_solver_options,
            )
            ##### TEMP for extracting intermediate outputs #####
            # self.odefunc.after_odeint()
            ####################################################

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        reg_states = state_t[2+addl_states:]

        if logpx is not None:
            return z_t, logpz_t, reg_states
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
