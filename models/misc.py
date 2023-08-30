import math
import torch
import torch.nn as nn


class PositionalEnc(nn.Module):
    def __init__(self, dim_in, seqlength, mode='sin'):
        super(PositionalEnc, self).__init__()
        if mode == 'sin':
            self.register_buffer('posenc', sinenc(dim_in, seqlength).view(1, seqlength, dim_in))
            self.register_buffer('logdetgrad', torch.zeros(1, 1).requires_grad_(False))
            self.pe = lambda x: (self.posenc, self.logdetgrad)
        elif mode == 'gru':
            raise NotImplementedError
        elif mode == 'floater':
            raise NotImplementedError

    def forward(self, x, context=None, logpx=None, integration_times=None, reverse=False):
        posenc, logdetgrad = self.pe(x)
        if reverse:
            if logpx is None:
                return x
            else:
                return x - posenc, logpx + logdetgrad
        else:
            if logpx is None:
                return x
            else:
                return x + posenc, logpx - logdetgrad


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, full_atom_feature_dims):
        super(AtomEncoder, self).__init__()

        atom_embedding_list = []

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            atom_embedding_list.append(emb)

        self.atom_embedding_list = torch.nn.ModuleList(atom_embedding_list)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim, full_bond_feature_dims):
        super(BondEncoder, self).__init__()

        bond_embedding_list = []

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            bond_embedding_list.append(emb)

        self.bond_embedding_list = torch.nn.ModuleList(bond_embedding_list)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


def sinenc(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        # raise ValueError("Cannot use sin/cos positional encoding with "
        #                  "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model+1)
    else:
        pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe[:, :d_model]
