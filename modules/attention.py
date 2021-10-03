import torch
from torch import nn
import math


class Attention(nn.Module):
    """Attention for graph """
    def __init__(self, input_dimension, out_dimension, n_head=1, use_bias=True):
        super(Attention, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = out_dimension
        self.n_head = n_head
        self.weight = nn.Parameter(torch.FloatTensor(input_dimension, out_dimension))
        self.a = nn.Parameter(torch.FloatTensor(2*out_dimension))
        self.leakyReLU = torch.nn.LeakyReLU(0.2)

        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dimension))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_features, neighbors_features):
        # print(node_features.shape, neighbors_features.shape)
        if len(node_features) > 0:
            return node_features
        else:
            sum_exp_eij = 0

            h_i = torch.FloatTensor(self.output_dimension)
            for neighbor_features in neighbors_features:
                e_i = torch.matmul(self.weight, node_features)
                e_j = torch.matmul(self.weight, neighbor_features)
                e_ij = torch.matmul(self.a, torch.cat((e_i, e_j), 0))
                exp_eij = math.exp(self.leakyReLU(e_ij))
            sum_exp_eij += exp_eij

            for neighbor_features in neighbors_features:
                e_i = torch.matmul(self.weight, node_features)
                e_j = torch.matmul(self.weight, neighbor_features)
                e_ij = torch.matmul(self.a, torch.cat((e_i, e_j), 0))
                exp_eij = math.exp(self.leakyReLU(e_ij))
                h_i += (exp_eij/sum_exp_eij) * torch.matmul(self.weight, e_j)
            return nn.ReLU(h_i)