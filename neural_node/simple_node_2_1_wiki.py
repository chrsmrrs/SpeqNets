import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from graph_tool.all import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.datasets import WikipediaNetwork, Actor
from torch_geometric.nn import GCNConv
from torch_scatter import scatter


class PPI_2_1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(PPI_2_1, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "squirdrfekgtfl"

    @property
    def processed_file_names(self):
        return "PPtdI_2_1fletffgg"

    def download(self):
        pass

    def process(self):

        dataset = 'chameleon'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = WikipediaNetwork(path, dataset, geom_gcn_preprocess = True)
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

        for c, t in enumerate(tuple_graph.vertices()):
            print(c, tuple_graph.num_vertices())

            # Get underlying nodes.
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

        data_new.train_mask = data.train_mask
        data_new.test_mask = data.test_mask
        data_new.val_mask = data.val_mask

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


path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ttertee')
dataset = PPI_2_1(path, transform=MyTransform())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 256

        self.mlp_init = Sequential(Linear(4652, dim), ReLU(), Linear(dim, dim))

        self.conv_1_1 = GCNConv(dim, dim)
        self.conv_1_2 = GCNConv(dim, dim)



        self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.conv_2_1 = GCNConv(dim, dim)
        self.conv_2_2 = GCNConv(dim, dim)
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.mlp = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, 5))

    def forward(self):
        x, edge_index_1, edge_index_2 = data.x, data.edge_index_1, data.edge_index_2

        index_1, index_2 = data.index_1, data.index_2

        x = self.mlp_init(x)

        x_1 = F.relu(self.conv_1_1(x, edge_index_1))
        x_2 = F.relu(self.conv_1_2(x, edge_index_2))
        x = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv_2_1(x, edge_index_1))
        x_2 = F.relu(self.conv_2_2(x, edge_index_2))
        x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

        index_1 = index_1.to(torch.int64)
        index_2 = index_2.to(torch.int64)
        x_1 = scatter(x, index_1, dim=0, reduce="mean")
        x_2 = scatter(x, index_2, dim=0, reduce="mean")
        x = self.mlp(torch.cat([x_1, x_2], dim=1))

        return F.log_softmax(x, dim=1)


def train(i):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask[:, i]], data.y[data.train_mask[:, i]]).backward()

    optimizer.step()


@torch.no_grad()
def test(i):
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask[:, i]].max(1)[1]
        acc = pred.eq(data.y[mask[:, i]]).sum().item() / mask[:, i].sum().item()
        accs.append(acc)
    return accs


acc_all = []
for i in range(1):
    acc_total = 0
    for i in range(10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, data = Net().to(device), data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

        best_val_acc = test_acc = 0
        for epoch in range(1, 201):
            train(i)
            train_acc, val_acc, tmp_test_acc = test(i)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            print(i, f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
                     f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

        acc_total += test_acc

    acc_all.append(acc_total/10)

print(np.array(acc_all).mean(), np.array(acc_all).std())
