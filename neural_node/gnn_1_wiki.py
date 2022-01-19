import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Twitch
from torch_geometric.nn import GCNConv, ChebConv  # noqa


dataset = 'squirrel'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Twitch(path, name="PT")


data = dataset[0]

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)



    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)



def train(i):
    model.train()
    optimizer.zero_grad()

    F.nll_loss(model()[data.train_mask[:,i]], data.y[data.train_mask[:,i]]).backward()
    optimizer.step()


@torch.no_grad()
def test(i):
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):

        pred = logits[mask[:,i]].max(1)[1]
        acc = pred.eq(data.y[mask[:,i]]).sum().item() / mask[:,i].sum().item()
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
print(dataset.num_classes)