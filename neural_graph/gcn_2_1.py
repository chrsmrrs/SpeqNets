import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from graph_tool.all import *
from sklearn.metrics import f1_score
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

from typing import Union, Optional

from torch import Tensor
from torch_sparse import SparseTensor

split = ""


class PPI_2_1(InMemoryDataset):
    def __init__(self, split, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(PPI_2_1, self).__init__( root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.split = split

    @property
    def raw_file_names(self):
        return "PPI_2_1"

    @property
    def processed_file_names(self):
        return "PPI_2_1"

    def download(self):
        pass

    def process(self):

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "PPI")
        dataset = PPI(path, split=split)
        data = dataset[0]

        x = data.x.cpu().detach().numpy()
        edge_index = data.edge_index.cpu().detach().numpy()

        # Create graph for easier processing.
        g = Graph(directed=False)
        num_nodes = x.shape[0]

        node_features = {}
        for i in range(num_nodes):
            v = g.add_vertex()
            node_features[v] = x[i]

        rows = list(edge_index[0])
        cols = list(edge_index[1])

        for ind, (i, j) in enumerate(zip(rows, cols)):
            g.add_edge(i, j, add_missing=False)

        tuple_graph = Graph(directed=False)
        type = {}

        tuple_to_nodes = {}
        nodes_to_tuple = {}
        for v in g.vertices():
            for w in v.all_neighbors():
                n = tuple_graph.add_vertex()
                tuple_to_nodes[n] = (v, w)
                nodes_to_tuple[(v, w)] = n

                type[n] = np.concatenate(
                    [node_features[v], node_features[w], np.array([1, 0])], axis=-1)

            n = tuple_graph.add_vertex()
            tuple_to_nodes[n] = (v, v)
            tuple_to_nodes[(v, v)] = n
            type[n] = np.concatenate([node_features[v], node_features[v], np.array([0, 1])], axis=-1)

        matrix_1 = []
        matrix_2 = []
        node_features = []

        index_1 = []
        index_2 = []

        for t in tuple_graph.vertices():
            v, w = tuple_to_nodes[t]

            node_features.append(type[t])
            index_1.append(int(v))
            index_2.append(int(w))

            # 1 neighbors.
            for n in v.out_neighbors():
                if (n, w) in nodes_to_tuple:
                    s = nodes_to_tuple[(n, w)]
                    matrix_1.append([int(t), int(s)])

            # 2 neighbors.
            for n in w.out_neighbors():
                if (v, n) in nodes_to_tuple:
                    s = nodes_to_tuple[(v, n)]
                    matrix_2.append([int(t), int(s)])

        data_list = []

        data_new = Data()

        edge_index_1 = torch.tensor(matrix_1).t().contiguous()
        edge_index_2 = torch.tensor(matrix_2).t().contiguous()

        data_new.edge_index_1 = edge_index_1
        data_new.edge_index_2 = edge_index_2

        data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)
        data_new.index_1 = torch.from_numpy(np.array(index_1)).to(torch.int64)
        data_new.index_2 = torch.from_numpy(np.array(index_2)).to(torch.int64)

        data_new.y = data.y

        data_list.append(data_new)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in [
            'edge_index_1', 'edge_index_2', 'index_1', 'index_2'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'eere')
split = "train"
train_dataset = PPI_2_1("train", path, transform=MyTransform())
split = "val"
val_dataset = PPI_2_1("val", path, transform=MyTransform())
split = "test"
test_dataset = PPI_2_1("test", path, transform=MyTransform())

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 256

        self.conv_1_1 = GINConv(
            Sequential(Linear(102, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.conv_1_2 = GINConv(
            Sequential(Linear(102, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.conv_2_1 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.conv_2_2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.conv_3_1 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.conv_3_2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.mlp = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, 121)


    def forward(self, x, edge_index_1, edge_index_2, index_1, index_2):

        x_1 = F.relu(self.conv_1_1(x, edge_index_1))
        x_2 = F.relu(self.conv_1_2(x, edge_index_2))
        x = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv_2_1(x, edge_index_1))
        x_2 = F.relu(self.conv_2_2(x, edge_index_2))
        x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv_3_1(x, edge_index_1))
        x_2 = F.relu(self.conv_3_2(x, edge_index_2))
        x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))


        index_1 = index_1.to(torch.int64)
        index_2 = index_2.to(torch.int64)
        x_1 = scatter(x, index_1, dim=0, reduce="mean")
        x_2 = scatter(x, index_2, dim=0, reduce="mean")

        x = self.mlp(torch.cat([x_1, x_2], dim=1))

        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        print(x.size())
        exit()

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()

    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data.x.to(device), data.edge_index_1.to(device), data.edge_index_2.to(device), data.index_1.to(device), data.index_2.to(device)), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index_1.to(device), data.edge_index_2.to(device), data.index_1.to(device), data.index_2.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for epoch in range(1, 2001):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))
