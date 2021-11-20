import sys

sys.path.insert(0, '..')
sys.path.insert(0, '.')

import aux as dp

import os.path as osp
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, GINConv
from graph_tool.all import *
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import pickle


class TUD_2_1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(TUD_2_1, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "TUD_2_1"

    @property
    def processed_file_names(self):
        return "TUD_2_1"

    def download(self):
        pass

    def process(self):
        data_list = []

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', "ZINC_test")
        dataset = TUDataset(path, name="ZINC_test").shuffle()

        data_list = []
        for i, data in enumerate(dataset):

            print(i)
            x = data.x.cpu().detach().numpy()
            edge_attr = data.edge_attr.cpu().detach().numpy()
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
            edge_features = {}

            for ind, (i, j) in enumerate(zip(rows, cols)):
                e = g.add_edge(i, j, add_missing=False)
                edge_features[e] = edge_attr[ind]

            type = {}

            tuple_to_nodes = {}
            nodes_to_tuple = {}

            tuples = 0
            for v in g.vertices():
                for w in v.all_neighbors():
                    n = tuples
                    tuples += 1

                    tuple_to_nodes[n] = (v, w)
                    nodes_to_tuple[(v, w)] = n

                    type[n] = np.concatenate(
                        [node_features[v], node_features[w], edge_features[g.edge(v, w)], np.array([1, 0])], axis=-1)

                n = tuples
                tuples += 1

                tuple_to_nodes[n] = (v, v)
                tuple_to_nodes[(v, v)] = n
                type[n] = np.concatenate([node_features[v], node_features[v], [0.0] * 3, np.array([0, 1])], axis=-1)

            matrix_1 = []
            matrix_2 = []
            node_features_t = []

            for t in range(tuples):
                v, w = tuple_to_nodes[t]

                node_features_t.append(type[t])

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

            data_new = Data()

            data_new.edge_index_1 = torch.tensor(matrix_1).t().contiguous().to(torch.long)
            data_new.edge_index_2 = torch.tensor(matrix_2).t().contiguous().to(torch.long)

            data_new.x = torch.from_numpy(np.array(node_features_t)).to(torch.float)
            data_new.y = data.y

            data_list.append(data_new)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in [
            'edge_index_1', 'edge_index_2'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


class NetGIN(torch.nn.Module):
    def __init__(self, dim):
        super(NetGIN, self).__init__()

        num_features = 97

        nn1_1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        nn1_2 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1_1 = GINConv(nn1_1, train_eps=True)
        self.conv1_2 = GINConv(nn1_2, train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        nn2_1 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        nn2_2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2_1 = GINConv(nn2_1, train_eps=True)
        self.conv2_2 = GINConv(nn2_2, train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        nn3_1 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        nn3_2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3_1 = GINConv(nn3_1, train_eps=True)
        self.conv3_2 = GINConv(nn3_2, train_eps=True)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        self.mlp_3 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        nn4_1 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        nn4_2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4_1 = GINConv(nn4_1, train_eps=True)
        self.conv4_2 = GINConv(nn4_2, train_eps=True)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        self.mlp_4 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.fc1 = Linear(4 * dim, dim)
        self.fc2 = Linear(dim, dim)
        self.fc3 = Linear(dim, dim)
        self.fc4 = Linear(dim, 1)

    def forward(self, data):
        x = data.x


        x_1 = F.relu(self.conv1_1(x, data.edge_index_1))
        x_2 = F.relu(self.conv1_2(x, data.edge_index_2))
        x_1_r = self.mlp_1(torch.cat([x_1, x_2], dim=-1))
        x_1_r = self.bn1(x_1_r)

        x_1 = F.relu(self.conv2_1(x_1_r, data.edge_index_1))
        x_2 = F.relu(self.conv2_2(x_1_r, data.edge_index_2))
        x_2_r = self.mlp_2(torch.cat([x_1, x_2], dim=-1))
        x_2_r = self.bn2(x_2_r)

        x_1 = F.relu(self.conv3_1(x_2_r, data.edge_index_1))
        x_2 = F.relu(self.conv3_2(x_2_r, data.edge_index_2))
        x_3_r = self.mlp_3(torch.cat([x_1, x_2], dim=-1))
        x_3_r = self.bn3(x_3_r)

        x_1 = F.relu(self.conv4_1(x_3_r, data.edge_index_1))
        x_2 = F.relu(self.conv4_2(x_3_r, data.edge_index_2))
        x_4_r = self.mlp_4(torch.cat([x_1, x_2], dim=-1))
        x_4_r = self.bn4(x_4_r)

        x = torch.cat([x_1_r, x_2_r, x_3_r, x_4_r], dim=-1)
        x = global_mean_pool(x, data.batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(-1)


plot_all = []
results = []

for _ in range(5):
    plot_it = []
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'testte')
    dataset = TUD_2_1(path, transform=MyTransform()).shuffle()

    train_dataset = dataset[0:4000]
    val_dataset = dataset[4000:4500]
    test_dataset = dataset[4500:]


    batch_size = 25
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NetGIN(256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=10,
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
        return loss_all / len(train_loader.dataset)


    def test(loader):
        model.eval()
        error = 0

        for data in loader:
            data = data.to(device)
            error += (model(data) - data.y).abs().sum().item()  # MAE
        return error / len(loader.dataset)


    best_val_error = None
    test_error = None
    for epoch in range(1, 1001):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error < best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error

        plot_it.append([loss, val_error, test_error])
        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

        if lr < 0.000001:
            print("Converged.")
            plot_all.append(plot_it)
            break

    results.append(test_error)

print("########################")
print(results)
results = np.array(results)
print(results.mean(), results.std())

