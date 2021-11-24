import sys

sys.path.insert(0, '..')
sys.path.insert(0, '.')

from torch_geometric.nn import GINConv
from graph_tool.all import *
from torch_geometric.datasets import TUDataset

import os.path as osp
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import Set2Set

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader

import torch
import torch.nn.functional as F

from aux import compute_k_s_tuple_graph_fast


class TUD_3_1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(TUD_3_1, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "TUD_2f_1gth"

    @property
    def processed_file_names(self):
        return "TUD_2f_g1th"

    def download(self):
        pass

    def process(self):
        atomic_type = {}
        atomic_counter = 0

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', "alchemy_full")
        dataset = TUDataset(path, name="alchemy_full")[0:20000]

        data_list = []
        for i, data in enumerate(dataset):
            print(i)

            x = data.x.cpu().detach().numpy()
            x = x.argmax(axis=-1)

            edge_attr = data.edge_attr.cpu().detach().numpy()
            edge_attr = edge_attr.argmax(axis=-1)

            edge_index = data.edge_index.cpu().detach().numpy()

            # Create graph for easier processing.
            g = Graph(directed=False)
            num_nodes = x.shape[0]

            node_labels = {}
            for i in range(num_nodes):
                v = g.add_vertex()
                node_labels[v] = x[i]

            rows = list(edge_index[0])
            cols = list(edge_index[1])
            edge_labels = {}

            for ind, (i, j) in enumerate(zip(rows, cols)):
                e = g.add_edge(i, j, add_missing=False)
                edge_labels[e] = edge_attr[ind]

            atomic_type, atomic_counter, matrices, labels = compute_k_s_tuple_graph_fast(g, 3, 1, node_labels,
                                                                                         edge_labels, atomic_type,
                                                                                         atomic_counter)

            data_new = Data()

            data_new.edge_index_1 = torch.tensor(matrices[0]).t().contiguous().to(torch.long)
            data_new.edge_index_2 = torch.tensor(matrices[1]).t().contiguous().to(torch.long)
            data_new.edge_index_3 = torch.tensor(matrices[2]).t().contiguous().to(torch.long)

            one_hot = np.eye(700)[labels]
            data_new.x = torch.from_numpy(one_hot).to(torch.float)

            data_new.y = data.y

            data_list.append(data_new)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in [
            'edge_index_1', 'edge_index_2', 'edge_index_3'
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

        num_features = 700

        nn1_1 = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn1_2 = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn1_3 = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv1_1 = GINConv(nn1_1, train_eps=True)
        self.conv1_2 = GINConv(nn1_2, train_eps=True)
        self.conv1_3 = GINConv(nn1_3, train_eps=True)
        self.mlp_1 = Sequential(Linear(3 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn2_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn2_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn2_3 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv2_1 = GINConv(nn2_1, train_eps=True)
        self.conv2_2 = GINConv(nn2_2, train_eps=True)
        self.conv2_3 = GINConv(nn2_3, train_eps=True)
        self.mlp_2 = Sequential(Linear(3 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn3_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn3_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn3_3 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv3_1 = GINConv(nn3_1, train_eps=True)
        self.conv3_2 = GINConv(nn3_2, train_eps=True)
        self.conv3_3 = GINConv(nn3_3, train_eps=True)
        self.mlp_3 = Sequential(Linear(3 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn4_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn4_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn4_3 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv4_1 = GINConv(nn4_1, train_eps=True)
        self.conv4_2 = GINConv(nn4_2, train_eps=True)
        self.conv4_3 = GINConv(nn4_3, train_eps=True)
        self.mlp_4 = Sequential(Linear(3 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn5_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn5_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn5_3 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv5_1 = GINConv(nn5_1, train_eps=True)
        self.conv5_2 = GINConv(nn5_2, train_eps=True)
        self.conv5_3 = GINConv(nn5_3, train_eps=True)
        self.mlp_5 = Sequential(Linear(3 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn6_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn6_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn6_3 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv6_1 = GINConv(nn6_1, train_eps=True)
        self.conv6_2 = GINConv(nn6_2, train_eps=True)
        self.conv6_3 = GINConv(nn6_3, train_eps=True)
        self.mlp_6 = Sequential(Linear(3 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())
        self.set2set = Set2Set(1 * dim, processing_steps=6)
        self.fc1 = Linear(2 * dim, dim)
        self.fc4 = Linear(dim, 12)

    def forward(self, data):
        x = data.x

        x_1 = F.relu(self.conv1_1(x, data.edge_index_1))
        x_2 = F.relu(self.conv1_2(x, data.edge_index_2))
        x_3 = F.relu(self.conv1_3(x, data.edge_index_3))
        x_1_r = self.mlp_1(torch.cat([x_1, x_2, x_3], dim=-1))

        x_1 = F.relu(self.conv2_1(x_1_r, data.edge_index_1))
        x_2 = F.relu(self.conv2_2(x_1_r, data.edge_index_2))
        x_3 = F.relu(self.conv2_3(x_1_r, data.edge_index_3))
        x_2_r = self.mlp_2(torch.cat([x_1, x_2, x_3], dim=-1))

        x_1 = F.relu(self.conv3_1(x_2_r, data.edge_index_1))
        x_2 = F.relu(self.conv3_2(x_2_r, data.edge_index_2))
        x_3 = F.relu(self.conv3_3(x_2_r, data.edge_index_3))
        x_3_r = self.mlp_3(torch.cat([x_1, x_2, x_3], dim=-1))

        x_1 = F.relu(self.conv4_1(x_3_r, data.edge_index_1))
        x_2 = F.relu(self.conv4_2(x_3_r, data.edge_index_2))
        x_3 = F.relu(self.conv4_3(x_3_r, data.edge_index_3))
        x_4_r = self.mlp_4(torch.cat([x_1, x_2, x_3], dim=-1))

        x_1 = F.relu(self.conv5_1(x_4_r, data.edge_index_1))
        x_2 = F.relu(self.conv5_2(x_4_r, data.edge_index_2))
        x_3 = F.relu(self.conv5_3(x_4_r, data.edge_index_3))
        x_5_r = self.mlp_5(torch.cat([x_1, x_2, x_3], dim=-1))

        x_1 = F.relu(self.conv6_1(x_5_r, data.edge_index_1))
        x_2 = F.relu(self.conv6_2(x_5_r, data.edge_index_2))
        x_3 = F.relu(self.conv6_3(x_5_r, data.edge_index_3))
        x_6_r = self.mlp_6(torch.cat([x_1, x_2, x_3], dim=-1))

        x = x_6_r

        x = self.set2set(x, data.batch)

        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x


plot_all = []
results = []

for _ in range(5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_it = []
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'tetsttte')
    dataset = TUD_3_1(path, transform=MyTransform())

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    train_dataset = dataset[0:18000]
    val_dataset = dataset[18000:19000]
    test_dataset = dataset[19000:]

    batch_size = 25
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = NetGIN(64).to(device)
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
    for epoch in range(1, 1001):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error, _ = test(val_loader)

        scheduler.step(val_error)
        if best_val_error is None or val_error <= best_val_error:
            test_error, test_error_log = test(test_loader)
            best_val_error = val_error

        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f},Test MAE: {:.7f}, '.format(epoch, lr, loss, val_error, test_error, test_error_log))

        if lr < 0.000001:
            print("Converged.")
            break
