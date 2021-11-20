import sys

sys.path.insert(0, '..')
sys.path.insert(0, '.')

import os.path as osp
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, MessagePassing

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F

import numpy as np

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


class NetGIN(torch.nn.Module):
    def __init__(self, dim):
        super(NetGIN, self).__init__()

        num_features = 6

        self.conv1_1 = GINConv(3, num_features, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2_1 = GINConv(3, dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3_1 = GINConv(3, dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4_1 = GINConv(3, dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(4 * dim, dim)
        self.fc2 = Linear(dim, dim)
        self.fc3 = Linear(dim, dim)
        self.fc4 = Linear(dim, 2)

    def forward(self, data):
        x = data.x
        edge_attr = data.edge_attr

        x_1 = F.relu(self.conv1_1(x, data.edge_index, edge_attr))
        x_1_r = self.bn1(x_1)

        x_2 = F.relu(self.conv2_1(x_1_r, data.edge_index, edge_attr))
        x_2_r = self.bn2(x_2)

        x_3 = F.relu(self.conv3_1(x_2_r, data.edge_index, edge_attr))
        x_3_r = self.bn3(x_3)

        x_4 = F.relu(self.conv4_1(x_3_r, data.edge_index, edge_attr))
        x_4_r = self.bn4(x_4)

        x = torch.cat([x_1_r, x_2_r, x_3_r, x_4_r], dim=-1)
        x = global_mean_pool(x, data.batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=-1)


plot_all = []
results = []

for _ in range(5):
    plot_it = []
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', "ZINC_test")
    dataset = TUDataset(path, name="ZINC_test").shuffle()

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

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_correct = 0
        for data in loader:
            data = data.to(device)
            out = model(data)
            total_correct += int((out.argmax(-1) == data.y).sum())
        return total_correct / len(loader.dataset)


    best_val_error = None
    test_error = None
    for epoch in range(1, 1001):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error > best_val_error:
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
