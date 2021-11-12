import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv
from torch_geometric.data import (InMemoryDataset, Data)
from graph_tool.all import *

class Cora(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(Cora, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "CORA21"

    @property
    def processed_file_names(self):
        return "CORA21"

    def download(self):
        pass

    def process(self):
        dataset = 'Cora'
        transform = T.Compose([
            T.RandomNodeSplit(num_val=500, num_test=500),
            T.TargetIndegree(),
        ])
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = Planetoid(path, dataset, transform=transform)
        data = dataset[0]

        data_list = []

        data_new = Data()
        data_new.edge_index = data.edge_index
        data_new.x = data.x
        data_new .edge_attr = data.edge_attr

        g = Graph(directed=False)
        num_nodes = data.x.size(-1)

        g.ep.edge_features = g.new_edge_property("vector<float>")
        for i  in range(num_nodes):
            g.add_vertex()

        rows = list(data.edge_index[0])
        cols = list(data.edge_index[1])

        g.ep.edge_features = g.new_edge_property("vector<float>")

        for ind, (i,j) in enumerate(zip(rows, cols)):
            e = g.add_edge(i.item(),j.item())
            #g.ep[e] = data.edge_attr[ind].cpu().detach().numpy()
            print(data.edge_attr[ind].cpu().detach().numpy())
            print(data.edge_attr[ind].cpu().detach().numpy().shape)
            print(type(data.edge_attr[ind].cpu().detach().numpy()))

        exit()

        data_list.append(data_new)

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


path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ZINC')
dataset = Cora(path, transform=MyTransform())
data = dataset[0]


exit()


dataset = 'Cora'
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
        self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
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