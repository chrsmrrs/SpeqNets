import sys

sys.path.insert(0, '..')
sys.path.insert(0, '.')
from timeit import default_timer as timer
import os.path as osp
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, MessagePassing

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F

import numpy as np

import os.path as osp
import numpy as np
from torch.nn import Linear as Lin
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import Set2Set

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader

import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

class GINConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv, self).__init__(aggr="add")

        self.bond_encoder = Sequential(Linear(emb_dim, dim1), ReLU(), Linear(dim1, dim1))
        self.mlp = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim2))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv, self).__init__(aggr="add")

        self.bond_encoder = Sequential(Linear(emb_dim, dim1), ReLU(), Linear(dim1, dim1))
        self.mlp = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim2))

        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class NetGINE(torch.nn.Module):
    def __init__(self, dim):
        super(NetGINE, self).__init__()

        num_features = 6
        dim = dim

        self.conv1 = GINConv(4, num_features, dim)
        self.conv2 = GINConv(4, dim, dim)
        self.conv3 = GINConv(4, dim, dim)
        self.conv4 = GINConv(4, dim, dim)
        self.conv5 = GINConv(4, dim, dim)
        self.conv6 = GINConv(4, dim, dim)

        self.set2set = Set2Set(1 * dim, processing_steps=6)

        self.fc1 = Lin(2 * dim, dim)
        self.fc4 = Linear(dim, 12)

    def forward(self, data):
        x = data.x

        x_1 = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x_2 = F.relu(self.conv2(x_1, data.edge_index, data.edge_attr))
        x_3 = F.relu(self.conv3(x_2, data.edge_index, data.edge_attr))
        x_4 = F.relu(self.conv4(x_3, data.edge_index, data.edge_attr))
        x_5 = F.relu(self.conv5(x_4, data.edge_index, data.edge_attr))
        x_6 = F.relu(self.conv6(x_5, data.edge_index, data.edge_attr))
        x = x_6
        x = self.set2set(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plot_it = []
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'datasets', "alchemy_full")

infile = open("train_al_10.index", "r")
for line in infile:
    indices_train = line.split(",")
    indices_train = [int(i) for i in indices_train]

infile = open("val_al_10.index", "r")
for line in infile:
    indices_val = line.split(",")
    indices_val = [int(i) for i in indices_val]

infile = open("test_al_10.index", "r")
for line in infile:
    indices_test = line.split(",")
    indices_test = [int(i) for i in indices_test]

indices = indices_train
indices.extend(indices_val)
indices.extend(indices_test)

dataset = TUDataset(path, name="alchemy_full")[indices]
print(len(dataset))

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.to(device), std.to(device)

train_dataset = dataset[0:10000]
val_dataset = dataset[10000:11000]
test_dataset = dataset[11000:]

batch_size = 25
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = NetGINE(64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=5,
                                                       min_lr=0.0000001)


def train():
    model.train()
    loss_all = 0

    lf = torch.nn.L1Loss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = lf(model(data), data.y)

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return (loss_all / len(train_loader.dataset))


@torch.no_grad()
def test(loader):
    model.eval()
    error = torch.zeros([1, 12]).to(device)

    for data in loader:
        data = data.to(device)
        error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

    error = error / len(loader.dataset)
    error_log = torch.log(error)

    return error.mean().item(), error_log.mean().item()

best_val_error = None
i = 0
torch.cuda.synchronize()
start = timer()
for epoch in range(1, 201):
    i += 1
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train()
    val_error, _ = test(val_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error, _ = test(test_loader)
        best_val_error = val_error

    print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

    if lr < 0.000001:
        print("Converged.")
        break
torch.cuda.synchronize()
stop = timer()
delta = stop - start
print(delta, delta / i)



