import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GCNConv,
    DiffGroupNorm
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter


class GATGNN_GIM1_globalATTENTION(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(GATGNN_GIM1_globalATTENTION, self).__init__()

        self.act = act
        self.fc_layers = fc_layers
        self.batch_track_stats = batch_track_stats == "True"
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.global_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()

        assert fc_layers > 1, "Need at least 2 fc layers"

        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim + 108, dim)
                self.global_mlp.append(lin)
            else:
                if i != self.fc_layers:
                    lin = torch.nn.Linear(dim, dim)
                else:
                    lin = torch.nn.Linear(dim, 1)
                self.global_mlp.append(lin)

            if self.batch_norm == "True":
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

    def forward(self, x, batch, glbl_x):
        out = torch.cat([x, glbl_x], dim=-1)
        for i in range(0, len(self.global_mlp)):
            if i != len(self.global_mlp) - 1:
                out = self.global_mlp[i](out)
                out = getattr(F, self.act)(out)
            else:
                out = self.global_mlp[i](out)
                out = tg_softmax(out, batch)
        return out


class GATGNN_AGAT_LAYER(MessagePassing):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2, **kwargs):
        super(GATGNN_AGAT_LAYER, self).__init__(aggr='add', flow='target_to_source', **kwargs)

        self.act = act
        self.fc_layers = fc_layers
        self.batch_track_stats = batch_track_stats == "True"
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.heads = 4
        self.add_bias = True
        self.neg_slope = 0.2

        self.bn1 = nn.BatchNorm1d(self.heads)
        self.W = Parameter(torch.Tensor(dim * 2, self.heads * dim))
        self.att = Parameter(torch.Tensor(1, self.heads, 2 * dim))
        self.dim = dim

        if self.add_bias:
            self.bias = Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        out_i = torch.cat([x_i, edge_attr], dim=-1)
        out_j = torch.cat([x_j, edge_attr], dim=-1)

        out_i = getattr(F, self.act)(torch.matmul(out_i, self.W))
        out_j = getattr(F, self.act)(torch.matmul(out_j, self.W))
        out_i = out_i.view(-1, self.heads, self.dim)
        out_j = out_j.view(-1, self.heads, self.dim)

        alpha = getattr(F, self.act)((torch.cat([out_i, out_j], dim=-1) * self.att).sum(dim=-1))
        alpha = getattr(F, self.act)(self.bn1(alpha))
        alpha = tg_softmax(alpha, edge_index_i)

        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        out_j = (out_j * alpha.view(-1, self.heads, 1)).transpose(0, 1)
        return out_j

    def update(self, aggr_out):
        out = aggr_out.mean(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out


class DEEP_GATGNN(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=5,
        post_fc_count=1,
        pool="global_add_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="softplus",
        dropout_rate=0.0,
        **kwargs
    ):
        super(DEEP_GATGNN, self).__init__()

        self.batch_track_stats = batch_track_stats == "True"
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate

        self.heads = 4
        self.global_att_LAYER = GATGNN_GIM1_globalATTENTION(dim1, act, batch_norm, batch_track_stats, dropout_rate)

        assert gc_count > 0, "Need at least 1 GNN layer"
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1

        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1

        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])

        if pre_fc_count > 0:
            self.pre_lin_list_E = torch.nn.ModuleList()
            self.pre_lin_list_N = torch.nn.ModuleList()

            for i in range(pre_fc_count):
                if i == 0:
                    lin_N = torch.nn.Linear(data.num_features, dim1)
                    self.pre_lin_list_N.append(lin_N)
                    lin_E = torch.nn.Linear(data.num_edge_features, dim1)
                    self.pre_lin_list_E.append(lin_E)
                else:
                    lin_N = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list_N.append(lin_N)
                    lin_E = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list_E.append(lin_E)

        elif pre_fc_count == 0:
            self.pre_lin_list_N = torch.nn.ModuleList()
            self.pre_lin_list_E = torch.nn.ModuleList()

        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = GATGNN_AGAT_LAYER(dim1, act, batch_norm, batch_track_stats, dropout_rate)
            self.conv_list.append(conv)
            if self.batch_norm == "True":
                bn = DiffGroupNorm(gc_dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim * 2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)

        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)

    def forward(self, data):
        for i in range(0, len(self.pre_lin_list_N)):
            if i == 0:
                out_x = self.pre_lin_list_N[i](data.x)
                out_x = getattr(F, 'leaky_relu')(out_x, 0.2)
                out_e = self.pre_lin_list_E[i](data.edge_attr)
                out_e = getattr(F, 'leaky_relu')(out_e, 0.2)
            else:
                out_x = self.pre_lin_list_N[i](out_x)
                out_x = getattr(F, self.act)(out_x)
                out_e = self.pre_lin_list_E[i](out_e)
                out_e = getattr(F, 'leaky_relu')(out_e, 0.2)
        prev_out_x = out_x

        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list_N) == 0 and i == 0:
                if self.batch_norm == "True":
                    out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    out_x = self.bn_list[i](out_x)
                else:
                    out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
            else:
                if self.batch_norm == "True":
                    out_x = self.conv_list[i](out_x, data.edge_index, out_e)
                    out_x = self.bn_list[i](out_x)
                else:
                    out_x = self.conv_list[i](out_x, data.edge_index, out_e)
            out_x = torch.add(out_x, prev_out_x)
            out_x = F.dropout(out_x, p=self.dropout_rate, training=self.training)
            prev_out_x = out_x

        out_a = self.global_att_LAYER(out_x, data.batch, data.glob_feat)
        out_x = out_x * out_a

        if self.pool_order == "early":
            if self.pool == "set2set":
                out_x = self.set2set(out_x, data.batch)
            else:
                out_x = getattr(torch_geometric.nn, self.pool)(out_x, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out_x = self.post_lin_list[i](out_x)
                out_x = getattr(F, self.act)(out_x)
            out = self.lin_out(out_x)

        elif self.pool_order == "late":
            for i in range(0, len(self.post_lin_list)):
                out_x = self.post_lin_list[i](out_x)
                out_x = getattr(F, self.act)(out_x)
            out = self.lin_out(out_x)
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out

    def get_embedding(self, data):
        # 获取嵌入向量，不影响训练过程
        self.eval()
        with torch.no_grad():
            for i in range(0, len(self.pre_lin_list_N)):
                if i == 0:
                    out_x = self.pre_lin_list_N[i](data.x)
                    out_x = getattr(F, 'leaky_relu')(out_x, 0.2)
                    out_e = self.pre_lin_list_E[i](data.edge_attr)
                    out_e = getattr(F, 'leaky_relu')(out_e, 0.2)
                else:
                    out_x = self.pre_lin_list_N[i](out_x)
                    out_x = getattr(F, self.act)(out_x)
                    out_e = self.pre_lin_list_E[i](out_e)
                    out_e = getattr(F, 'leaky_relu')(out_e, 0.2)
            prev_out_x = out_x

            for i in range(0, len(self.conv_list)):
                if len(self.pre_lin_list_N) == 0 and i == 0:
                    if self.batch_norm == "True":
                        out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                        out_x = self.bn_list[i](out_x)
                    else:
                        out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                else:
                    if self.batch_norm == "True":
                        out_x = self.conv_list[i](out_x, data.edge_index, out_e)
                        out_x = self.bn_list[i](out_x)
                    else:
                        out_x = self.conv_list[i](out_x, data.edge_index, out_e)
                out_x = torch.add(out_x, prev_out_x)
                out_x = F.dropout(out_x, p=self.dropout_rate, training=self.training)
                prev_out_x = out_x

            out_a = self.global_att_LAYER(out_x, data.batch, data.glob_feat)
            out_x = out_x * out_a

            if self.pool_order == "early":
                if self.pool == "set2set":
                    out_x = self.set2set(out_x, data.batch)
                else:
                    out_x = getattr(torch_geometric.nn, self.pool)(out_x, data.batch)

            elif self.pool_order == "late":
                if self.pool == "set2set":
                    out_x = self.set2set(out_x, data.batch)
                else:
                    out_x = getattr(torch_geometric.nn, self.pool)(out_x, data.batch)

            return out_x
