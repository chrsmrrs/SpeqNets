from __future__ import division

import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')



import auxiliarymethods.datasets as dp
import preprocessing as pre



import os.path as osp
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, Set2Set
import numpy as np

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
import torch.nn.functional as F

class QM9(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "QM9_2trs"


    @property
    def processed_file_names(self):
        return "QM9_2trs"

    def download(self):
        pass

    def process(self):
        data_list = []
        targets = dp.get_dataset("QM9", multigregression=True).tolist()

        attributes = pre.get_all_attributes_2_1("QM9")

        node_labels = pre.get_all_node_labels_2_1("QM9", False, False)
        matrices = pre.get_all_matrices_2_1_malkin("QM9", list(range(129433)))

        for i, m in enumerate(matrices):
            edge_index_1_l = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_1_g = torch.tensor(matrices[i][1]).t().contiguous()
            edge_index_2_l = torch.tensor(matrices[i][2]).t().contiguous()
            edge_index_2_g = torch.tensor(matrices[i][3]).t().contiguous()

            data = Data()
            data.edge_index_1_l = edge_index_1_l
            data.edge_index_1_g = edge_index_1_g
            data.edge_index_2_l = edge_index_2_l
            data.edge_index_2_g = edge_index_2_g

            one_hot = np.eye(3)[node_labels[i]]
            data.x = torch.from_numpy(one_hot).to(torch.float)

            # Continuous information.
            data.first = torch.from_numpy(np.array(attributes[i][0])[:,0:13]).to(torch.float)
            data.first_coord = torch.from_numpy(np.array(attributes[i][0])[:, 13:]).to(torch.float)

            data.second = torch.from_numpy(np.array(attributes[i][1])[:,0:13]).to(torch.float)
            data.second_coord = torch.from_numpy(np.array(attributes[i][1])[:, 13:]).to(torch.float)
            data.dist = torch.norm(data.first_coord - data.second_coord, p=2, dim=-1).view(-1, 1)
            data.edge_attr = torch.from_numpy(np.array(attributes[i][2])).to(torch.float)
            data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in [
            'edge_index_1_l', 'edge_index_1_g', 'edge_index_2_l', 'edge_index_2_g'
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

        self.node_attribute_encoder = Sequential(Linear(2*13, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.type_encoder = Sequential(Linear(3, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                       torch.nn.BatchNorm1d(dim), ReLU())
        self.edge_encoder = Sequential(Linear(4+1, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.mlp = Sequential(Linear(3*dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())

        nn1_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn1_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn1_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn1_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())

        self.conv1_1_l = GINConv(nn1_1_l, train_eps=True)
        self.conv1_2_l = GINConv(nn1_2_l, train_eps=True)
        self.conv1_1_g = GINConv(nn1_1_g, train_eps=True)
        self.conv1_2_g = GINConv(nn1_2_g, train_eps=True)

        self.mlp_1 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn2_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn2_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn2_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn2_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())

        self.conv2_1_l = GINConv(nn2_1_l, train_eps=True)
        self.conv2_2_l = GINConv(nn2_2_l, train_eps=True)
        self.conv2_1_g = GINConv(nn2_1_g, train_eps=True)
        self.conv2_2_g = GINConv(nn2_2_g, train_eps=True)

        self.mlp_2 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn3_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn3_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn3_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn3_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv3_1_l = GINConv(nn3_1_l, train_eps=True)
        self.conv3_2_l = GINConv(nn3_2_l, train_eps=True)
        self.conv3_1_g = GINConv(nn3_1_g, train_eps=True)
        self.conv3_2_g = GINConv(nn3_2_g, train_eps=True)

        self.mlp_3 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn4_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn4_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn4_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn4_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv4_1_l = GINConv(nn4_1_l, train_eps=True)
        self.conv4_2_l = GINConv(nn4_2_l, train_eps=True)
        self.conv4_1_g = GINConv(nn4_1_g, train_eps=True)
        self.conv4_2_g = GINConv(nn4_2_g, train_eps=True)

        self.mlp_4 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn5_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn5_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn5_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn5_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())

        self.conv5_1_l = GINConv(nn5_1_l, train_eps=True)
        self.conv5_2_l = GINConv(nn5_2_l, train_eps=True)
        self.conv5_1_g = GINConv(nn5_1_g, train_eps=True)
        self.conv5_2_g = GINConv(nn5_2_g, train_eps=True)

        self.mlp_5 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn6_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn6_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn6_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn6_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())

        self.conv6_1_l = GINConv(nn6_1_l, train_eps=True)
        self.conv6_2_l = GINConv(nn6_2_l, train_eps=True)
        self.conv6_1_g = GINConv(nn6_1_g, train_eps=True)
        self.conv6_2_g = GINConv(nn6_2_g, train_eps=True)

        self.mlp_6 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())
        self.set2set = Set2Set(1 * dim, processing_steps=6)
        self.fc1 = Linear(2 * dim, dim)
        self.fc4 = Linear(dim, 12)

    def forward(self, data):
        first, second, edge_attr, dist =  data.first, data.second, data.edge_attr, data.dist

        node_labels = data.x
        node_labels = self.type_encoder(node_labels)

        node_attributes = torch.cat([first, second], dim=-1)
        node_attributes = self.node_attribute_encoder(node_attributes)

        edge_attributes = torch.cat([edge_attr, dist], dim=-1)
        edge_attributes = self.edge_encoder(edge_attributes)


        x = torch.cat([node_labels, node_attributes, edge_attributes], dim=-1)
        x = self.mlp(x)

        x_1 = F.relu(self.conv1_1_l(x, data.edge_index_1_l))
        x_2 = F.relu(self.conv1_2_l(x, data.edge_index_2_l))
        x_3 = F.relu(self.conv1_1_g(x, data.edge_index_1_g))
        x_4 = F.relu(self.conv1_2_g(x, data.edge_index_2_g))
        x_1_r = self.mlp_1(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv2_1_l(x_1_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv2_2_l(x_1_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv2_1_g(x_1_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv2_2_g(x_1_r, data.edge_index_2_g))
        x_2_r = self.mlp_2(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv3_1_l(x_2_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv3_2_l(x_2_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv3_1_g(x_2_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv3_2_g(x_2_r, data.edge_index_2_g))
        x_3_r = self.mlp_3(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv4_1_l(x_3_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv4_2_l(x_3_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv4_1_g(x_3_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv4_2_g(x_3_r, data.edge_index_2_g))
        x_4_r = self.mlp_4(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv5_1_l(x_4_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv5_2_l(x_4_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv5_1_g(x_4_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv5_2_g(x_4_r, data.edge_index_2_g))
        x_5_r = self.mlp_5(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv6_1_l(x_5_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv6_2_l(x_5_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv6_1_g(x_5_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv6_2_g(x_5_r, data.edge_index_2_g))
        x_6_r = self.mlp_6(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x = x_6_r

        x = self.set2set(x, data.batch)

        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x


results = []
results_log = []
for _ in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'QM9')
    dataset = QM9(path, transform=MyTransform()).shuffle()
    dataset.data.y = dataset.data.y[:,0:12]

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    tenpercent = int(len(dataset) * 0.1)
    test_dataset = dataset[:tenpercent].shuffle()
    val_dataset = dataset[tenpercent:2 * tenpercent].shuffle()
    train_dataset = dataset[2 * tenpercent:].shuffle()

    batch_size = 64
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

    test_error = None
    test_error_log = None
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
              'Test MAE: {:.7f}, Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error, test_error_log))

        if lr < 0.000001:
            print("Converged.")
            break

    results.append(test_error)
    results_log.append(test_error_log)

print("########################")
print(results)
results = np.array(results)
print(results.mean(), results.std())

print(results_log)
results_log = np.array(results_log)
print(results_log.mean(), results_log.std())
