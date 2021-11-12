import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from graph_tool.all import *
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SplineConv


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

        # Create graph for easier processing.
        g = Graph(directed=False)
        num_nodes = data.x.size(-1)


        node_features = {}
        for i in range(num_nodes):
            v = g.add_vertex()
            node_features[v] = data.x[i].cpu().detach().numpy()
            print(node_features[v].shape)

        exit()

        rows = list(data.edge_index[0])
        cols = list(data.edge_index[1])
        g.ep.edge_features = g.new_edge_property("double")

        for ind, (i, j) in enumerate(zip(rows, cols)):
            e = g.add_edge(i.item(), j.item())
            g.ep.edge_features[e] = data.edge_attr[ind].item()

        tuple_graph = Graph(directed=False)
        type = {}

        tuple_to_nodes = {}
        nodes_to_tuple = {}
        for v in g.vertices():
            for w in v.out_neighbors():
                n = tuple_graph.add_vertex()
                tuple_to_nodes[n] = (v, w)
                nodes_to_tuple[(v, w)] = n

                print(node_features[v].shape)
                print(node_features[w].shape)
                print("###")

                type[n] = np.concatenate([node_features[v], node_features[w]], axis=-1)


                print(type[n].shape)

            n = tuple_graph.add_vertex()
            tuple_to_nodes[n] = (v, v)
            tuple_to_nodes[(v, v)] = n
            type[n] = np.concatenate([node_features[v], node_features[v]])
            #print(np.array(tuple_graph.vp.type[n]).shape)


        exit()

        matrix_1 = []
        matrix_2 = []
        node_features = []

        for t in tuple_graph.vertices():
            v, w = tuple_to_nodes[t]

            node_features.append(type[t])

            #print(np.array(tuple_graph.vp.type[t]).shape)

            # 1 neighbors.
            for n in v.out_neighbors():
                if (n, w) in nodes_to_tuple:
                    s = nodes_to_tuple[(n, w)]
                    e = tuple_graph.add_edge(t, s)
                    tuple_graph.ep.edge_features[e] = 1

                    matrix_1.append([int(t), int(s)])

            # 2 neighbors.
            for n in w.out_neighbors():
                if (v, n) in nodes_to_tuple:
                    s = nodes_to_tuple[(v, n)]
                    e = tuple_graph.add_edge(t, s)
                    tuple_graph.ep.edge_features[e] = 2

                    matrix_2.append([int(t), int(s)])

        data_list = []

        data_new = Data()

        edge_index_1 = torch.tensor(matrix_1).t().contiguous()
        edge_index_2 = torch.tensor(matrix_2).t().contiguous()

        data_new.edge_index_1 = edge_index_1
        data_new.edge_index_2 = edge_index_2

        for n in node_features:
            print(n.shape)

        exit()

        data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)
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
