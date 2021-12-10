import math
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


#nets
class TDA_FCNet(nn.Module):
    def __init__(self, in_dim, tda_dim, n_classes):
        super().__init__()
        shared_out_dim = 30
        self.fc1 = nn.Linear(in_dim, 256) #in, out
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, shared_out_dim)
        self.tda_fc1 = nn.Linear(tda_dim, 64)
        self.tda_fc2 = nn.Linear(64, 64)
        self.tda_fc3 = nn.Linear(64, shared_out_dim)
        self.fc4 = nn.Linear(shared_out_dim, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc_last =  nn.Linear(64, n_classes)
        self.in_dim = in_dim
        self.tda_dim = tda_dim

    def forward(self, x, x_tda):
        #print("x size, xtda size", x.size(), x_tda.size())
        #x, x_tda = torch.split(x, [self.in_dim, self.tda_dim])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x_tda = F.relu(self.tda_fc1(x_tda))
        x_tda = F.relu(self.tda_fc2(x_tda))
        x_tda = F.relu(self.tda_fc3(x_tda))
        x = x + x_tda #with/out tda
        x =  F.relu(self.fc4(x))
        x =  F.relu(self.fc5(x))
        x =  F.relu(self.fc6(x))
        x =  F.relu(self.fc7(x))
        x = self.fc_last(x)
        return x

#the working architecture
class TDA_GCNet_padded(nn.Module):
    def __init__(self, in_dim, adj_dim, tda_dim, n_classes, diag=False, concat=False, with_tda=False):
        super().__init__()
        #shared_out_dim = 30
        self.concat = concat
        gcn_hidden_dim = 64
        gcn_out_dim = 30
        self.gc1 = GraphConvolution(in_dim, gcn_hidden_dim) #base gc1 gc2
        self.gc2 = GraphConvolution(gcn_hidden_dim, gcn_hidden_dim)
        self.gc3 = GraphConvolution(gcn_hidden_dim, gcn_hidden_dim) #deep gc3 gc4
        self.gc4 = GraphConvolution(gcn_hidden_dim, gcn_out_dim)
        if concat:
            dim = gcn_out_dim + gcn_hidden_dim*3 #if concat
        else:
            dim = gcn_out_dim
        self.fc1 =  nn.Linear(adj_dim*(dim), 64)   
        self.fc2 =  nn.Linear(64, 64)
        self.fc3 =  nn.Linear(64, 64)
        self.fc1_tda = nn.Linear(tda_dim, 32)
        self.fc2_tda = nn.Linear(32, 64)
        self.fc_last =  nn.Linear(64, n_classes)
        self.in_dim = in_dim
        self.tda_dim = tda_dim
        self.adj_dim = adj_dim
        self.diag = diag
        self.with_tda = with_tda

    def forward(self, x, x_tda, adj, D=None):
        if self.diag and D is not None:
            x1 = F.relu(self.gc1(x, adj, D))
            x2 = F.relu(self.gc2(x1, adj, D))
            x3 = F.relu(self.gc3(x2, adj, D))
            x4 = F.relu(self.gc4(x3, adj, D))
        else:
            x1 = F.relu(self.gc1(x, adj))
            x2 = F.relu(self.gc2(x1, adj))
            x3 = F.relu(self.gc3(x2, adj))
            x4 = F.relu(self.gc4(x3, adj))
        if self.concat:
            x = torch.cat((x4,x3,x2,x1), dim=2) 
        else:
            x = x4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if self.with_tda:
            x_tda = F.relu(self.fc1_tda(x_tda))
            x_tda = F.relu(self.fc2_tda(x_tda))
            x = x + x_tda #with tda
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_last(x)
        #return F.log_softmax(x, dim=1)
        return x

class TDA_GCNet(nn.Module):
    def __init__(self, in_dim, adj_dim, tda_dim, n_classes, diag=False, concat=False):
        super().__init__()
        self.slice_ind = 128
        #shared_out_dim = 30
        self.concat = concat
        gcn_hidden_dim = 64
        gcn_out_dim = 30
        self.gc1 = GraphConvolution(in_dim, gcn_hidden_dim) #base gc1 gc2
        self.gc2 = GraphConvolution(gcn_hidden_dim, gcn_hidden_dim)
        self.gc3 = GraphConvolution(gcn_hidden_dim, gcn_hidden_dim) #deep gc3 gc4
        self.gc4 = GraphConvolution(gcn_hidden_dim, gcn_out_dim)
        if concat:
            dim = gcn_out_dim + gcn_hidden_dim*3 #if concat
        else:
            dim = gcn_out_dim
        self.fc1 =  nn.Linear(self.slice_ind, 64)   
        self.fc2 =  nn.Linear(64, 64)
        self.fc3 =  nn.Linear(64, 64)
        self.fc1_tda = nn.Linear(tda_dim, 32)
        self.fc2_tda = nn.Linear(32, 64)
        self.fc_last =  nn.Linear(64, n_classes)
        self.in_dim = in_dim
        self.tda_dim = tda_dim
        self.adj_dim = adj_dim
        self.diag = diag

    def forward(self, x, x_tda, adj, D=None):
        if self.diag and D is not None:
            x1 = F.relu(self.gc1(x, adj, D))
            x2 = F.relu(self.gc2(x1, adj, D))
            x3 = F.relu(self.gc3(x2, adj, D))
            x4 = F.relu(self.gc4(x3, adj, D))
        else:
            x1 = F.relu(self.gc1(x, adj))
            x2 = F.relu(self.gc2(x1, adj))
            x3 = F.relu(self.gc3(x2, adj))
            x4 = F.relu(self.gc4(x3, adj))
        if self.concat:
            x = torch.cat((x4,x3,x2,x1), dim=2) 
        else:
            x = torch.cat((x4,x4),dim=2)
        x = torch.flatten(x, 1)
        x = x[:,:self.slice_ind]
        x_tda = F.relu(self.fc1_tda(x_tda))
        x_tda = F.relu(self.fc2_tda(x_tda))
        x = F.relu(self.fc1(x))
        x = x + x_tda #with tda
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_last(x)
        #return F.log_softmax(x, dim=1)
        return x

#Datasets
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class TDADataset(torch.utils.data.Dataset):
    def __init__(self, X, X_tda, Y):
        self.X = X.astype(np.float32)
        self.X_tda = X_tda.astype(np.float32)
        self.Y = Y.astype(np.float32)
        print(self.Y.shape)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.X_tda[index], self.Y[index]


class TDA_GCN_Dataset(torch.utils.data.Dataset):
    def __init__(self, X_feat, X_adj, X_tda, Y, diag=False):
        self.X_feat = X_feat.astype(np.float32)
        print("TDA GCN DATASET ")
        print(self.X_feat.shape)
        self.X_feat = np.expand_dims(self.X_feat, 2)
        print(self.X_feat.shape)
        self.X_adj = X_adj.astype(np.float32)
        self.X_tda = X_tda.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.diag = diag
        if diag:
            self.D = list(self.X_adj)
            self.D = [ np.diag(np.reciprocal(np.sum(xi,1)+1e-5)) for xi in self.D ] 
            self.D = np.array(self.D).astype(np.float32)


    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):
        if self.diag:
            return self.X_feat[index], self.X_tda[index], self.X_adj[index], self.D[index], self.Y[index]     
        return self.X_feat[index], self.X_tda[index], self.X_adj[index], self.Y[index]


class TDA_GCN_Dataset_no_pad(torch.utils.data.Dataset):
    def __init__(self, X_feat, X_adj, X_tda, Y, diag=False):
        self.X_feat = [x.astype(np.float32) for x in X_feat]
        print("TDA GCN DATASET ")
        #self.X_feat = [np.expand_dims(x, 1).astype(np.float32) for x in X_feat]
        
        self.X_adj = [x.astype(np.float32) for x in X_adj]
        self.X_tda =[x.astype(np.float32) for x in X_tda]
        self.Y = Y.astype(np.float32)
        self.diag = diag
        if diag:
            self.D = list(self.X_adj)
            self.D = [ np.diag(np.reciprocal(np.sum(xi,1))).astype(np.float32) for xi in self.D ] 


    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):
        if self.diag:
            return self.X_feat[index], self.X_tda[index], self.X_adj[index], self.D[index], self.Y[index]     
        return self.X_feat[index], self.X_tda[index], self.X_adj[index], self.Y[index]

#other- adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, D=None):
        #print("inp, weight, adj, size",  input.size(), self.weight.size(), adj.size())
        support = torch.matmul(input, self.weight)
        #print("sup adj, size", support.size(), adj.size())
        output = torch.matmul(adj, support)
        #print("OUT,", output.size())
        if D is not None:
            output = torch.matmul(D, output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'