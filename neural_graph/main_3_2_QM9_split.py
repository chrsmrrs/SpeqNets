from __future__ import division

import sys

import auxiliarymethods.datasets as dp
import preprocessing as pre

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os.path as osp
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, Set2Set
import numpy as np

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
import torch.nn.functional as F

class QM9_1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9_1, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "QM9_2trttt"


    @property
    def processed_file_names(self):
        return "QM9_2trttt"

    def download(self):
        pass

    def process(self):
        data_list = []
        targets = dp.get_dataset("QM9", multigregression=True).tolist()
        attributes = pre.get_all_attributes_3_2("QM9", list(range(30000)))

        node_labels = pre.get_all_node_labels_3_2("QM9", False, False, list(range(30000)))
        matrices = pre.get_all_matrices_3_2("QM9", list(range(30000)))

        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()
            edge_index_3 = torch.tensor(matrices[i][2]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2
            data.edge_index_3 = edge_index_3

            # one_hot = np.eye(50)[node_labels[i]]
            # data.x = torch.from_numpy(one_hot).to(torch.float)
            data.x = torch.from_numpy(np.array(node_labels[i])).to(torch.float)

            # Continuous information.
            data.first = torch.from_numpy(np.array(attributes[i][0])[:,0:13]).to(torch.float)
            data.first_coord = torch.from_numpy(np.array(attributes[i][0])[:, 13:]).to(torch.float)

            data.second = torch.from_numpy(np.array(attributes[i][1])[:, 0:13]).to(torch.float)
            data.second_coord = torch.from_numpy(np.array(attributes[i][1])[:, 13:]).to(torch.float)

            data.third = torch.from_numpy(np.array(attributes[i][2])[:, 0:13]).to(torch.float)
            data.third_coord = torch.from_numpy(np.array(attributes[i][2])[:, 13:]).to(torch.float)

            data.dist_12 = torch.norm(data.first_coord - data.second_coord, p=2, dim=-1).view(-1, 1)
            data.dist_13 = torch.norm(data.first_coord - data.third_coord, p=2, dim=-1).view(-1, 1)
            data.dist_23 = torch.norm(data.second_coord - data.third_coord, p=2, dim=-1).view(-1, 1)

            data.edge_attr = torch.from_numpy(np.array(attributes[i][3])).to(torch.float)
            data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class QM9_2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9_2, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "QM9_2trtt55t"


    @property
    def processed_file_names(self):
        return "QM9_2ttrt55t"

    def download(self):
        pass

    def process(self):
        data_list = []
        targets = dp.get_dataset("QM9", multigregression=True).tolist()
        attributes = pre.get_all_attributes_3_2("QM9", list(range(30000, 60000)))

        node_labels = pre.get_all_node_labels_3_2("QM9", False, False, list(range(30000, 60000)))
        matrices = pre.get_all_matrices_3_2("QM9", list(range(30000, 60000)))

        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()
            edge_index_3 = torch.tensor(matrices[i][2]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2
            data.edge_index_3 = edge_index_3

            # one_hot = np.eye(50)[node_labels[i]]
            # data.x = torch.from_numpy(one_hot).to(torch.float)
            data.x = torch.from_numpy(np.array(node_labels[i])).to(torch.float)

            # Continuous information.
            data.first = torch.from_numpy(np.array(attributes[i][0])[:,0:13]).to(torch.float)
            data.first_coord = torch.from_numpy(np.array(attributes[i][0])[:, 13:]).to(torch.float)

            data.second = torch.from_numpy(np.array(attributes[i][1])[:, 0:13]).to(torch.float)
            data.second_coord = torch.from_numpy(np.array(attributes[i][1])[:, 13:]).to(torch.float)

            data.third = torch.from_numpy(np.array(attributes[i][2])[:, 0:13]).to(torch.float)
            data.third_coord = torch.from_numpy(np.array(attributes[i][2])[:, 13:]).to(torch.float)

            data.dist_12 = torch.norm(data.first_coord - data.second_coord, p=2, dim=-1).view(-1, 1)
            data.dist_13 = torch.norm(data.first_coord - data.third_coord, p=2, dim=-1).view(-1, 1)
            data.dist_23 = torch.norm(data.second_coord - data.third_coord, p=2, dim=-1).view(-1, 1)

            data.edge_attr = torch.from_numpy(np.array(attributes[i][3])).to(torch.float)
            data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class QM9_3(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9_3, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "QM9_2tttt77"


    @property
    def processed_file_names(self):
        return "QM9_2t77ttt"

    def download(self):
        pass

    def process(self):
        data_list = []
        targets = dp.get_dataset("QM9", multigregression=True).tolist()
        attributes = pre.get_all_attributes_3_2("QM9", list(range(60000, 90000)))

        node_labels = pre.get_all_node_labels_3_2("QM9", False, False, list(range(60000, 90000)))
        matrices = pre.get_all_matrices_3_2("QM9", list(range(60000, 90000)))

        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()
            edge_index_3 = torch.tensor(matrices[i][2]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2
            data.edge_index_3 = edge_index_3

            # one_hot = np.eye(50)[node_labels[i]]
            # data.x = torch.from_numpy(one_hot).to(torch.float)
            data.x = torch.from_numpy(np.array(node_labels[i])).to(torch.float)

            # Continuous information.
            data.first = torch.from_numpy(np.array(attributes[i][0])[:,0:13]).to(torch.float)
            data.first_coord = torch.from_numpy(np.array(attributes[i][0])[:, 13:]).to(torch.float)

            data.second = torch.from_numpy(np.array(attributes[i][1])[:, 0:13]).to(torch.float)
            data.second_coord = torch.from_numpy(np.array(attributes[i][1])[:, 13:]).to(torch.float)

            data.third = torch.from_numpy(np.array(attributes[i][2])[:, 0:13]).to(torch.float)
            data.third_coord = torch.from_numpy(np.array(attributes[i][2])[:, 13:]).to(torch.float)

            data.dist_12 = torch.norm(data.first_coord - data.second_coord, p=2, dim=-1).view(-1, 1)
            data.dist_13 = torch.norm(data.first_coord - data.third_coord, p=2, dim=-1).view(-1, 1)
            data.dist_23 = torch.norm(data.second_coord - data.third_coord, p=2, dim=-1).view(-1, 1)

            data.edge_attr = torch.from_numpy(np.array(attributes[i][3])).to(torch.float)
            data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class QM9_4(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9_4, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "QM966_2tttt"


    @property
    def processed_file_names(self):
        return "QM966_2tttt"

    def download(self):
        pass

    def process(self):
        data_list = []
        targets = dp.get_dataset("QM9", multigregression=True).tolist()
        attributes = pre.get_all_attributes_3_2("QM9", list(range(90000, 129433)))

        node_labels = pre.get_all_node_labels_3_2("QM9", False, False, list(range(90000, 129433)))
        matrices = pre.get_all_matrices_3_2("QM9", list(range(90000, 129433)))

        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()
            edge_index_3 = torch.tensor(matrices[i][2]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2
            data.edge_index_3 = edge_index_3

            # one_hot = np.eye(50)[node_labels[i]]
            # data.x = torch.from_numpy(one_hot).to(torch.float)
            data.x = torch.from_numpy(np.array(node_labels[i])).to(torch.float)

            # Continuous information.
            data.first = torch.from_numpy(np.array(attributes[i][0])[:,0:13]).to(torch.float)
            data.first_coord = torch.from_numpy(np.array(attributes[i][0])[:, 13:]).to(torch.float)

            data.second = torch.from_numpy(np.array(attributes[i][1])[:, 0:13]).to(torch.float)
            data.second_coord = torch.from_numpy(np.array(attributes[i][1])[:, 13:]).to(torch.float)

            data.third = torch.from_numpy(np.array(attributes[i][2])[:, 0:13]).to(torch.float)
            data.third_coord = torch.from_numpy(np.array(attributes[i][2])[:, 13:]).to(torch.float)

            data.dist_12 = torch.norm(data.first_coord - data.second_coord, p=2, dim=-1).view(-1, 1)
            data.dist_13 = torch.norm(data.first_coord - data.third_coord, p=2, dim=-1).view(-1, 1)
            data.dist_23 = torch.norm(data.second_coord - data.third_coord, p=2, dim=-1).view(-1, 1)

            data.edge_attr = torch.from_numpy(np.array(attributes[i][3])).to(torch.float)
            data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class QM9_all(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9_all, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZIgNC_trairn_all"

    @property
    def processed_file_names(self):
        return "ZIgNC_trrain_all"

    def download(self):
        pass

    def process(self):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'QM9')

        # TODO?? no transform
        dataset_1 = QM9_1(path)
        dataset_2 = QM9_2(path)
        dataset_3 = QM9_3(path)
        dataset_4 = QM9_4(path)

        dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4])
        data_list = []

        for i,data in enumerate(dataset):
            print(i)
            data_list.append(data)

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

        self.node_attribute_encoder = Sequential(Linear(3*13, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.type_encoder = Sequential(Linear(14, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                       torch.nn.BatchNorm1d(dim), ReLU())
        self.edge_encoder = Sequential(Linear(15, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.mlp = Sequential(Linear(3*dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())

        nn1_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn1_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv1_1 = GINConv(nn1_1, train_eps=True)
        self.conv1_2 = GINConv(nn1_2, train_eps=True)
        self.mlp_1 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn2_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn2_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv2_1 = GINConv(nn2_1, train_eps=True)
        self.conv2_2 = GINConv(nn2_2, train_eps=True)
        self.mlp_2 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn3_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn3_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv3_1 = GINConv(nn3_1, train_eps=True)
        self.conv3_2 = GINConv(nn3_2, train_eps=True)
        self.mlp_3 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn4_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn4_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv4_1 = GINConv(nn4_1, train_eps=True)
        self.conv4_2 = GINConv(nn4_2, train_eps=True)
        self.mlp_4 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn5_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn5_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv5_1 = GINConv(nn5_1, train_eps=True)
        self.conv5_2 = GINConv(nn5_2, train_eps=True)
        self.mlp_5 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn6_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn6_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv6_1 = GINConv(nn6_1, train_eps=True)
        self.conv6_2 = GINConv(nn6_2, train_eps=True)
        self.mlp_6 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())
        self.set2set = Set2Set(1 * dim, processing_steps=6)
        self.fc1 = Linear(2 * dim, dim)
        self.fc4 = Linear(dim, 12)

    def forward(self, data):
        first, second, third, edge_attr =  data.first, data.second, data.third, data.edge_attr

        dist_12 = data.dist_12
        dist_13 = data.dist_13
        dist_23 = data.dist_23

        x = data.x
        x = x.long()

        node_labels = torch.zeros(x.size(0), 14).to(device)
        node_labels[range(node_labels.shape[0]), x.view(1, x.size(0))] = 1

        node_labels = self.type_encoder(node_labels)

        node_attributes = torch.cat([first, second, third], dim=-1)
        node_attributes = self.node_attribute_encoder(node_attributes)

        edge_attributes = torch.cat([edge_attr, dist_12, dist_13, dist_23], dim=-1)

        edge_attributes = self.edge_encoder(edge_attributes)



        print(node_labels.size(), node_attributes.size(), edge_attributes.size())
        exit()

        x = torch.cat([node_labels, node_attributes, edge_attributes], dim=-1)
        x = self.mlp(x)

        x_1 = F.relu(self.conv1_1(x, data.edge_index_1))
        x_2 = F.relu(self.conv1_2(x, data.edge_index_2))
        x_1_r = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv2_1(x_1_r, data.edge_index_1))
        x_2 = F.relu(self.conv2_2(x_1_r, data.edge_index_2))
        x_2_r = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv3_1(x_2_r, data.edge_index_1))
        x_2 = F.relu(self.conv3_2(x_2_r, data.edge_index_2))
        x_3_r = self.mlp_3(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv4_1(x_3_r, data.edge_index_1))
        x_2 = F.relu(self.conv4_2(x_3_r, data.edge_index_2))
        x_4_r = self.mlp_4(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv5_1(x_4_r, data.edge_index_1))
        x_2 = F.relu(self.conv5_2(x_4_r, data.edge_index_2))
        x_5_r = self.mlp_5(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv6_1(x_5_r, data.edge_index_1))
        x_2 = F.relu(self.conv6_2(x_5_r, data.edge_index_2))
        x_6_r = self.mlp_6(torch.cat([x_1, x_2], dim=-1))

        x = x_6_r

        x = self.set2set(x, data.batch)

        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x



results = []
results_log = []
for _ in range(3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'QM9')
    dataset = QM9_all(path, transform=MyTransform()).shuffle()


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
