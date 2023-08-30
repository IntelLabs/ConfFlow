import torch
import copy
import torch.nn as nn
import torch.nn.init as init
from .graphprop import MPNN, NodeModel, PT_block, mlp_function


def weights_init(m):
    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight.data)
            if m.bias is not None:
                init.zeros_(m.bias.data)

    if isinstance(m, nn.ModuleList):
        for layer in m:
            layer.apply(weights_init)
    else:
        m.apply(_weight_init)


def weight_init_zeros(m):
    """
    This is similar as the function above where we initialize linear layers from a normal distribution with std
    1./10 as suggested by the author. This should only be used for the message passing functions, i.e. fe's in the
    paper.
    """

    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.zeros_(m.weight.data)
            if m.bias is not None:
                init.zeros_(m.bias.data)

    if isinstance(m, nn.ModuleList):
        for layer in m:
            layer.apply(weight_init_zeros)
    else:
        m.apply(_weight_init)


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_T=1):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, context, x):
        return context, self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_T=1):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + dim_T + dim_c, dim_out)

    def forward(self, context, x, c):
        if x.dim() == 3 and context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)
        x_context = torch.cat((x, context), dim=2)
        return context, self._layer(x_context)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_T=1):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_T + dim_c, dim_out, bias=False)

    def forward(self, context, x):
        bias = self._hyper_bias(context)
        if x.dim() == 3 and context.dim() == 2:
            bias = bias.unsqueeze(1)
        return context, self._layer(x) + bias


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_T=1):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(dim_T + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper(context))
        if x.dim() == 3 and context.dim() == 2:
            gate = gate.unsqueeze(1)
        return context, self._layer(x) * gate


class ScaleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_T=1):
        super(ScaleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(dim_T + dim_c, dim_out)

    def forward(self, context, x):
        gate = self._hyper(context)
        if x.dim() == 3 and context.dim() == 2:
            gate = gate.unsqueeze(1)
        return context, self._layer(x) * gate


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_T=1):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_T + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_T + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        if x.dim() == 3 and context.dim() == 2:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return context, ret


class ConcatScaleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_T=1):
        super(ConcatScaleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_T + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_T + dim_c, dim_out)

    def forward(self, context, x):
        gate = self._hyper_gate(context)
        bias = self._hyper_bias(context)
        if x.dim() == 3 and context.dim() == 2:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return context, ret


class MPNNPoint(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_T, **kwargs):
        super(MPNNPoint, self).__init__()

        self.in_dim = dim_in
        self.nblocks = 2
        self.act = copy.deepcopy(kwargs['act'])
        hidden_dim = [64, 128, dim_out]
        self.embedding_dim = hidden_dim[1]

        self.pe = mlp_function(self.embedding_dim + self.in_dim, hidden_dim[:-1], act=self.act)

        # Processing alongside edge attributes
        mpnn = list()
        mpnn.append(MPNN(self.embedding_dim, hidden_dim[:-2], is_megnet=False, act=self.act))
        for _ in range(self.nblocks-1):
            mpnn.append(MPNN(self.embedding_dim, hidden_dim[:-2], is_megnet=True, act=self.act))

        self.mpnn = nn.ModuleList(mpnn)

        # Post-processing for node positions
        self.out = NodeModel(hidden_dim[1:], in_dim=3 * self.embedding_dim + dim_c + dim_T, act=self.act)

    def forward(self, context, x):
        nodes, edges, virtual, edge_index, node_batch, lengths, t = \
            context[0], context[1], context[2], context[3], context[4], context[5], context[6]

        in_sz = virtual.size(0)
        nodes_x = torch.cat([nodes, x], dim=-1)
        nodes = self.pe(nodes_x)
        t1 = t.expand(in_sz, -1)

        for i in range(self.nblocks):
            nodes_out, edges_out, virtual_out = self.mpnn[i](nodes, edge_index, edges, virtual, node_batch)
            nodes = nodes + nodes_out
            edges = edges + edges_out
            virtual = virtual + virtual_out

        if len(context) == 8:
            ret = self.out(nodes, edge_index, edges, torch.cat([virtual, t1, context[7]], dim=-1), node_batch)

            return tuple([nodes, edges, virtual, edge_index, node_batch, lengths, t, context[7]]), ret
        else:
            ret = self.out(nodes, edge_index, edges, torch.cat([virtual, t1], dim=-1), node_batch)

            return tuple([nodes, edges, virtual, edge_index, node_batch, lengths, t]), ret


class PointTxFormer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_T, **kwargs):
        super(PointTxFormer, self).__init__()

        self.in_dim = dim_in
        hidden_dim = [64, 128, dim_out]
        self.embedding_dim = hidden_dim[1]

        self.nblocks = 2
        self.act = copy.deepcopy(kwargs['act'])

        # Processing alongside edge attributes
        self.mpnn = PT_block(self.nblocks, self.in_dim, self.embedding_dim, hidden_dim[:-2], dim_T, act=self.act)

        # Post-processing for node positions
        self.out = NodeModel(hidden_dim[1:], in_dim=3 * self.embedding_dim + dim_c + dim_T, act=self.act)

    def forward(self, context, x):
        nodes, edges, virtual, edge_index, node_batch, lengths, t = \
            context[0], context[1], context[2], context[3], context[4], context[5], context[6]

        in_sz = virtual.size(0)
        t1 = t.expand(in_sz, -1)

        nodes, edges, _ = self.mpnn(x, nodes, edge_index, edges, virtual, node_batch, t)

        if len(context) == 8:
            ret = self.out(nodes, edge_index, edges, torch.cat([virtual, context[7], t1], dim=-1), node_batch)

            return tuple([nodes, edges, virtual, edge_index, node_batch, lengths, t, context[7]]), ret
        else:
            ret = self.out(nodes, edge_index, edges, torch.cat([virtual, t1], dim=-1), node_batch)

            return tuple([nodes, edges, virtual, edge_index, node_batch, lengths, t]), ret
