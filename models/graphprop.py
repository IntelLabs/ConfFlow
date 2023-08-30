import torch
import torch.nn as nn
import copy
from torch_scatter import scatter_mean, scatter_add
from torch_scatter.composite import scatter_softmax


def clones(module, k):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(k)])


class mlp_function(nn.Module):
    def __init__(self, input_dim, hidden_dim, act, bias=True):
        super(mlp_function, self).__init__()
        dim = [input_dim] + hidden_dim

        layers = []
        for j, k in list(zip(dim[:-1], dim[1:])):
            layers.append(nn.Linear(j, k, bias=bias))
            layers.append(copy.deepcopy(act))
        layers = layers[:-1]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class EdgeModel(nn.Module):
    def __init__(self, hidden_dim, act):
        super(EdgeModel, self).__init__()
        self.edge_mlp = mlp_function(4 * hidden_dim[-1], hidden_dim, act=act)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, hidden_dim, act, in_dim=None, use_sent_edge=False):
        super(NodeModel, self).__init__()
        self.use_sent_edge = use_sent_edge
        if use_sent_edge:
            in_dim = 4 * hidden_dim[-1] if in_dim is None else in_dim
            self.node_mlp = mlp_function(in_dim, hidden_dim, act=act)
        else:
            in_dim = 3 * hidden_dim[-1] if in_dim is None else in_dim
            self.node_mlp = mlp_function(in_dim, hidden_dim, act=act)

    def forward(self, node_attr, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        list_attr = [node_attr]

        if self.use_sent_edge:
            list_attr.append(scatter_mean(edge_attr, row, dim=0, dim_size=node_attr.size(0)))

        list_attr.append(scatter_mean(edge_attr, col, dim=0, dim_size=node_attr.size(0)))

        list_attr.append(u[batch])

        out = torch.cat(list_attr, dim=1)
        return self.node_mlp(out)


class GlobalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, act):
        super(GlobalModel, self).__init__()
        self.global_mlp = mlp_function(input_dim, hidden_dim, act=act)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        list_attr = [u]

        list_attr.append(scatter_mean(x, batch, dim=0))

        list_attr.append(scatter_mean(edge_attr, batch[row], dim=0))

        out = torch.cat(list_attr, dim=1)
        return self.global_mlp(out)


class MetaLayer(nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, time=None):
        """"""
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u,
                                        batch if batch is None else batch[row])

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

        if self.global_model is not None:
            if time is not None:
                u = torch.cat([u, time], dim=-1)
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)


class MPNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, is_megnet=False,
                 dim_T=0, act=nn.ReLU(inplace=False)):
        super(MPNN, self).__init__()
        self.is_megnet = is_megnet
        dim = hidden_dim + [input_dim]

        edge = EdgeModel(dim, act=act)
        node = NodeModel(dim, act=act)
        virtual = GlobalModel(3 * input_dim + dim_T, dim, act=act)

        self.out = MetaLayer(edge, node, virtual)

        if self.is_megnet:
            dense = nn.Sequential(
                nn.Linear(input_dim, hidden_dim[0]),
                nn.Linear(hidden_dim[0], input_dim)
            )
            self.dense = clones(dense, 3)

    def forward(self, nodes_in, edge_index, edges_in, global_in, batch_index, t=None):
        if self.is_megnet:
            nodes = self.dense[0](nodes_in)
            edges = self.dense[1](edges_in)
            virtual = self.dense[2](global_in)
        else:
            nodes = nodes_in
            edges = edges_in
            virtual = global_in

        nodes_out, edges_out, global_out = self.out(nodes, edge_index, edge_attr=edges,
                                                    u=virtual, batch=batch_index, time=t)

        return nodes_out, edges_out, global_out


class PT(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, dim_T, act=nn.ReLU(inplace=False)):
        super(PT, self).__init__()
        hidden_dim = hidden_dim + [embedding_dim]

        self.varphi = nn.Linear(embedding_dim + dim_T, embedding_dim)
        self.phi = nn.Linear(embedding_dim + dim_T, embedding_dim)
        self.alpha = nn.Linear(embedding_dim + dim_T, embedding_dim)
        self.beta = mlp_function(2 * embedding_dim + dim_T, hidden_dim, act=act)
        self.gamma = mlp_function(2 * embedding_dim + dim_T, hidden_dim, act=act)
        self.delta = mlp_function(input_dim + dim_T, hidden_dim, act=act)

    def forward(self, x, nodes_in, edge_index, edges_in, global_in, batch_index, t=None):
        row, col = edge_index

        natoms = x.size(0)
        if t is None:
            t = torch.empty(natoms, 0).type_as(x)
        else:
            t = t.expand(natoms, -1)
        nodes = torch.cat([nodes_in, t], dim=-1)

        B = x[row] - x[col]
        delta = self.delta(torch.cat([B, t[row]], dim=-1))

        varphi = self.varphi(nodes)
        phi = self.phi(nodes)
        edges_out = self.gamma(torch.cat([varphi[row] - phi[col] + edges_in, delta, t[row]], dim=-1))

        alpha = self.alpha(nodes)
        rho = scatter_softmax(edges_out, row, dim=0)
        nodes_out = self.beta(torch.cat([scatter_add(rho * (alpha[col] + delta), row, dim=0, dim_size=natoms), nodes], dim=-1))

        return nodes_out, edges_out


class PT_block(nn.Module):
    def __init__(self, nblocks, input_dim, embedding_dim, hidden_dim, dim_T, act):
        super(PT_block, self).__init__()
        self.nblocks = nblocks

        self.mpnn = torch.nn.ModuleList()
        for _ in range(self.nblocks):
            self.mpnn.append(PT(input_dim, embedding_dim, hidden_dim, dim_T, act=act))

    def forward(self, x, nodes, edge_index, edges, virtual, node_batch, t=None):
        ret = x

        for i in range(self.nblocks):
            nodes_out, edges_out = self.mpnn[i](ret, nodes, edge_index, edges, virtual, node_batch, t)
            nodes = nodes_out + nodes
            edges = edges_out + edges

        return nodes, edges, ret


class GNF_MP(nn.Module):
    def __init__(self, nblocks, input_dim, output_dim, node_embedding_size, hidden_dim, act=nn.SiLU()):
        super(GNF_MP, self).__init__()
        dim_T = 0

        self.mpnn = PT_block(nblocks, input_dim, node_embedding_size, hidden_dim, dim_T, act=act)
        self.out = NodeModel([node_embedding_size, output_dim], in_dim=3 * node_embedding_size, act=act)

    def forward(self, x, context):
        nodes, edges, virtual, edge_index, node_batch = \
            context[0], context[1], context[2], context[3], context[4]

        nodes, edges, _ = self.mpnn(x, nodes, edge_index, edges, virtual, node_batch)
        ret = self.out(nodes, edge_index, edges, virtual, node_batch)

        return ret
