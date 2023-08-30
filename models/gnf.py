import numpy as np
from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.init as init
from .graphprop import GNF_MP, clones
from .diffeq_layers import weights_init, weight_init_zeros
from .normalization import MovingBatchNorm1d
from .odefunc import NONLINEARITIES


def gnn_message_weight_init(m, gain=1./10):
    def _weight_init(z):
        if isinstance(z, nn.Linear):
            init.normal_(z.weight.data, std=gain)
            if z.bias is not None:
                init.normal_(z.bias.data, std=gain)

    if isinstance(m, nn.ModuleList):
        for layer in m:
            layer.apply(gnn_message_weight_init)
    else:
        m.apply(_weight_init)


class GRevNet(nn.Module):
    def __init__(self, args, weight_sharing=False):
        super(GRevNet, self).__init__()
        self.weight_sharing = weight_sharing
        self.use_batch_norm = args.batch_norm
        self.num_prop_rounds = args.num_blocks
        self.nonlinearity = NONLINEARITIES[args.nonlinearity]

        self.nblocks = 2
        self.input_dim = args.input_dim
        self.node_embedding_size = 128
        self.hidden_dim = [64]
        # initial gain
        self.gain = 1. / np.sqrt(2*self.num_prop_rounds)

        self.graph_prop_1 = GNF_MP(self.nblocks, 1, 2, self.node_embedding_size, self.hidden_dim, self.nonlinearity)
        self.graph_prop_2 = GNF_MP(self.nblocks, 2, 1, self.node_embedding_size, self.hidden_dim, self.nonlinearity)

        if weight_sharing:
            s = [deepcopy(self.graph_prop_1), deepcopy(self.graph_prop_2)]
            t = [deepcopy(self.graph_prop_1), deepcopy(self.graph_prop_2)]
        else:
            s = [clones(self.graph_prop_1, self.num_prop_rounds), clones(self.graph_prop_2, self.num_prop_rounds)]
            t = [clones(self.graph_prop_1, self.num_prop_rounds), clones(self.graph_prop_2, self.num_prop_rounds)]

        # original decay = 0.005
        bn1 = [MovingBatchNorm1d(2, decay=0.1) for _ in range(self.num_prop_rounds + 1)]
        bn2 = [MovingBatchNorm1d(1, decay=0.1) for _ in range(self.num_prop_rounds + 1)]

        self.s = nn.ModuleList(s)
        self.t = nn.ModuleList(t)
        self.bns = nn.ModuleList([nn.ModuleList(bn1), nn.ModuleList(bn2)])

        self.init_weights()

    def init_weights(self):
        message_weight_init = partial(gnn_message_weight_init, gain=self.gain)

        self.s.apply(weights_init)
        self.t.apply(weights_init)
        if self.weight_sharing:
            for j in range(2):
                self.s[j].out.node_mlp.mlp[-1].apply(weight_init_zeros)
                self.t[j].out.node_mlp.mlp[-1].apply(weight_init_zeros)
                for k in range(self.nblocks):
                    self.s[j].mpnn.mpnn[k].varphi.apply(message_weight_init)
                    self.s[j].mpnn.mpnn[k].phi.apply(message_weight_init)
                    self.s[j].mpnn.mpnn[k].alpha.apply(message_weight_init)
                    self.s[j].mpnn.mpnn[k].beta.mlp.apply(message_weight_init)
                    self.s[j].mpnn.mpnn[k].gamma.mlp.apply(message_weight_init)
                    self.s[j].mpnn.mpnn[k].delta.mlp.apply(message_weight_init)
                    self.t[j].mpnn.mpnn[k].varphi.apply(message_weight_init)
                    self.t[j].mpnn.mpnn[k].phi.apply(message_weight_init)
                    self.t[j].mpnn.mpnn[k].alpha.apply(message_weight_init)
                    self.t[j].mpnn.mpnn[k].beta.mlp.apply(message_weight_init)
                    self.t[j].mpnn.mpnn[k].gamma.mlp.apply(message_weight_init)
                    self.t[j].mpnn.mpnn[k].delta.mlp.apply(message_weight_init)
        else:
            for j in range(2):
                for i in range(self.num_prop_rounds):
                    self.s[j][i].out.node_mlp.mlp[-1].apply(weight_init_zeros)
                    self.t[j][i].out.node_mlp.mlp[-1].apply(weight_init_zeros)
                    for k in range(self.nblocks):
                        self.s[j][i].mpnn.mpnn[k].varphi.apply(message_weight_init)
                        self.s[j][i].mpnn.mpnn[k].phi.apply(message_weight_init)
                        self.s[j][i].mpnn.mpnn[k].alpha.apply(message_weight_init)
                        self.s[j][i].mpnn.mpnn[k].beta.mlp.apply(message_weight_init)
                        self.s[j][i].mpnn.mpnn[k].gamma.mlp.apply(message_weight_init)
                        self.s[j][i].mpnn.mpnn[k].delta.mlp.apply(message_weight_init)
                        self.t[j][i].mpnn.mpnn[k].varphi.apply(message_weight_init)
                        self.t[j][i].mpnn.mpnn[k].phi.apply(message_weight_init)
                        self.t[j][i].mpnn.mpnn[k].alpha.apply(message_weight_init)
                        self.t[j][i].mpnn.mpnn[k].beta.mlp.apply(message_weight_init)
                        self.t[j][i].mpnn.mpnn[k].gamma.mlp.apply(message_weight_init)
                        self.t[j][i].mpnn.mpnn[k].delta.mlp.apply(message_weight_init)

    def forward_train(self, x, context, log_det_jacobian):
        x1 = x[:, :self.input_dim // 2]
        x2 = x[:, self.input_dim // 2:]

        if self.use_batch_norm:
            x2, log_det_jacobian, _ = self.bns[0][0](x2, logpx=log_det_jacobian)
            x1, log_det_jacobian, _ = self.bns[1][0](x1, logpx=log_det_jacobian)

        for i in range(self.num_prop_rounds):
            if self.weight_sharing:
                s = self.s[0](x1, context)
                t = self.t[0](x1, context)
            else:
                s = self.s[0][i](x1, context)
                t = self.t[0][i](x1, context)

            # negation is done to follow conventional code
            log_det_jacobian += -s.sum(dim=-1, keepdim=True)
            x2 = x2 * torch.exp(s) + t

            if self.use_batch_norm:
                x2, log_det_jacobian, _ = self.bns[0][i + 1](x2, logpx=log_det_jacobian)

            if self.weight_sharing:
                s = self.s[1](x2, context)
                t = self.t[1](x2, context)
            else:
                s = self.s[1][i](x2, context)
                t = self.t[1][i](x2, context)

            log_det_jacobian += -s.sum(dim=-1, keepdim=True)
            x1 = x1 * torch.exp(s) + t

            if self.use_batch_norm:
                x1, log_det_jacobian, _ = self.bns[1][i + 1](x1, logpx=log_det_jacobian)

        x = torch.cat([x1, x2], dim=-1)
        return x, log_det_jacobian

    def forward_inference(self, x, context):
        x1 = x[:, :self.input_dim // 2]
        x2 = x[:, self.input_dim // 2:]

        for i in reversed(range(self.num_prop_rounds)):
            if self.use_batch_norm:
                x1, _, _ = self.bns[1][i + 1](x1, reverse=True)

            if self.weight_sharing:
                s = self.s[1](x2, context)
                t = self.t[1](x2, context)
            else:
                s = self.s[1][i](x2, context)
                t = self.t[1][i](x2, context)

            x1 = (x1 - t) * torch.exp(-s)

            if self.use_batch_norm:
                x2, _, _ = self.bns[0][i + 1](x2, reverse=True)

            if self.weight_sharing:
                s = self.s[0](x1, context)
                t = self.t[0](x1, context)
            else:
                s = self.s[0][i](x1, context)
                t = self.t[0][i](x1, context)

            x2 = (x2 - t) * torch.exp(-s)

        if self.use_batch_norm:
            x2, _, _ = self.bns[0][0](x2, reverse=True)
            x1, _, _ = self.bns[1][0](x1, reverse=True)

        x = torch.cat([x1, x2], dim=-1)
        return x

    def forward(self, x, context, logpx, reverse=False):
        if reverse:
            return self.forward_inference(x, context)
        else:
            z, log_prob = self.forward_train(x, context, logpx)
            return z, log_prob
