import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50
import torch.nn.init as init
import torch_sparse
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter
from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes

num_atom_type = 119  # including the extra mask tokens
# 分子种类有119个
num_chirality_tag = 3
# 手性标签3个

num_bond_type = 5  # including aromatic and self-loop edge
# 包括自链接 键连接总共5种

num_bond_direction = 3


# 键连接方向3种

def gcn_norm(edge_index, num_nodes=None):
    # 定义gcn准则函数
    '''edge_index = tensor([[1, 9],
                            [9, 1]])'''
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    # 根据边关系存储, 返回可能的节点个数

    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    # edge_weight为一个长度为edge_index.size(1)=2 元素全为1的张量
    # tensor([1.,1.])

    row, col = edge_index[0], edge_index[1]
    # row=([1,9])
    # col=([9,1])

    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    # self.scatter_(dim, index, src, reduce)
    '''dim即维度，是对于self而言的，即在self的哪一dim进行操作 dim=0为行 dim=1为列
       index是索引，即要在self的哪一index进行操作
       index的维度可以小于等于src，如果二者维度相同，则相当于将src的每一个数字都加到self的对应index上
       如果index维度小，例如src: shape[5,3], index: shape[3,2]则代表只有src[:3,:2]的数字参与了操作
       src是待操作的源数字
       reduce代表操作的方式，none代表直接赋值，add则是+=，multiply是*=
       因此scatter的意思就是 将src中前index部分的数字以一定的方式scatter(散布)到self中

       dim=0时 src分布的行索引由index确定 列索引由自身分散元素所在列确定
       dim=1时 src分布的行索引由自身分散元素所在行确定 列索引由index确定'''

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    '''这个返回值应该可以理解为对于边权重的行列方向的变化'''


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.emb_dim = emb_dim
        # 获取内嵌维度为300

        self.aggr = aggr
        # 聚合方式为相加

        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        # Parameter 将不可训练的张量转化为可训练的参数类型，同时将转化后的张量绑定到模型可训练参数的列表中，当更新模型的参数时一并将其更新
        # self.weight为300x300的可求梯度的矩阵

        self.bias = Parameter(torch.Tensor(emb_dim))
        # self.bias为1x300的可求梯度的矩阵

        self.reset_parameters()
        # 参数初始化 但是我也不知道是对于哪些参数

        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        # nn.Embedding 创建一个简单的存储固定大小的词典的嵌入向量的查找表
        # self.edge_embedding1 词典尺寸为num_bond_type=5 内嵌向量维度为1

        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)
        # self.edge_embedding2 词典尺寸为num_bond_direction=3 内嵌向量维度为1

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_是一个服从均匀分布的Glorot初始化器
        # self.edge_embedding1
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        # 定义初始化函数
        # glorot(self.weight)
        # zeros(self.bias)
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        # stdv=sqrt(6.0/(300+300)) 即1/100

        self.weight.data.uniform_(-stdv, stdv)
        # self.weight均匀化处理(-stdv, stdv) 即(-0.1,0.1)

        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # 在边空间中添加自循环
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        # self_loop_attr =([[0., 0.]]))
        self_loop_attr[:, 0] = 4
        # bond type for self-loop edge
        ##self_loop_attr =([[0., 4.]]))

        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        edge_index, __ = gcn_norm(edge_index)

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr):
        # return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)


class DIGNN(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        # def __init__(self,args):
        super(DIGNN, self).__init__()
        '''self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio'''
        self.num_layer = 5
        self.emb_dim = 300
        self.feat_dim = 256
        self.drop_ratio = 0
        pool = 'mean'

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            self.gnns.append(GCNConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            # nn.Softplus(),
            nn.Linear(self.feat_dim, self.feat_dim // 2)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)

        return h, out


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class MERIT(nn.Module):
    def __init__(self, args, moving_average_decay):
        super().__init__()

        self.online_encoder = DIGNN(args)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self):
        # model1 = self.online_encoder(args)
        model2 = self.target_encoder
        return model2


if __name__ == "__main__":
    model = DIGNN()
    print(model)
