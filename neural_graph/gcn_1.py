import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch.nn import Sequential, Linear, ReLU
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

dataset = 'Planetoid'
transform = T.Compose([
    T.RandomNodeSplit(num_val=500, num_test=500),
    T.TargetIndegree(),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 64

        self.conv1 = GCNConv(dataset.num_features, dim)
        self.conv2 = GCNConv(dim, dim)

        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dataset.num_classes))

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))