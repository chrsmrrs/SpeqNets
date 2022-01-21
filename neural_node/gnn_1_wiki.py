import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Twitch, LastFMAsia, WikiCS, Planetoid
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from sklearn.model_selection import train_test_split



path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "feef")
dataset = Twitch(path, name="DE")
data = dataset[0]


print(data.x.size())


l = len(data.x)
l = list(range(l))
train, test = train_test_split(l, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)

train_mask = [True if i in train else False for i in l]
val_mask = [True if i in val else False for i in l]
test_mask = [True if i in test else False for i in l]

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
    F.nll_loss(model()[train_mask], data.y[train_mask]).backward()

    optimizer.step()



@torch.no_grad()
def test(i):
    model.eval()
    logits, accs = model(), []
    for mask in [train_mask, val_mask, test_mask]:


        pred = logits[mask[:]].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / sum(mask)
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

        acc_total += test_acc*100

    acc_all.append(acc_total/10)

print(np.array(acc_all).mean(), np.array(acc_all).std())

print(test_acc)
print(dataset.num_classes)