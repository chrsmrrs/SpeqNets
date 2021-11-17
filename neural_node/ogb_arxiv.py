import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from graph_tool.all import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_scatter import scatter

#from logger import Logger

class Arxiv(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(Arxiv, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "arxiv"

    @property
    def processed_file_names(self):
        return "arxiv"

    def download(self):
        pass

    def process(self):

        dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                         transform=T.ToSparseTensor())

        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()

        print(data.adj_t)
        print(data.adj_t[0])
        exit()

        x = data.x.cpu().detach().numpy()

        # Create graph for easier processing.
        g = Graph(directed=False)
        num_nodes = x.shape[0]

        node_features = {}
        for i in range(num_nodes):
            v = g.add_vertex()
            node_features[v] = x[i]

        rows = list(data.adj_t.row.cpu().detach())
        cols = list(data.adj_t.col.cpu().detach())
        g.ep.edge_features = g.new_edge_property("double")

        for ind, (i, j) in enumerate(zip(rows, cols)):
            e = g.add_edge(i, j, add_missing=False)
            g.ep.edge_features[e] = data.edge_attr[ind].item()

        tuple_graph = Graph(directed=False)
        type = {}

        tuple_to_nodes = {}
        nodes_to_tuple = {}
        for v in g.vertices():
            for w in v.all_neighbors():
                n = tuple_graph.add_vertex()
                tuple_to_nodes[n] = (v, w)
                nodes_to_tuple[(v, w)] = n

                type[n] = np.concatenate(
                    [node_features[v], node_features[w], [g.ep.edge_features[g.edge(v, w)]], np.array([1, 0])], axis=-1)

            n = tuple_graph.add_vertex()
            tuple_to_nodes[n] = (v, v)
            tuple_to_nodes[(v, v)] = n
            type[n] = np.concatenate([node_features[v], node_features[v], [0.0], np.array([0, 1])], axis=-1)

        matrix_1 = []
        matrix_2 = []
        node_features = []

        index_1 = []
        index_2 = []

        for t in tuple_graph.vertices():
            v, w = tuple_to_nodes[t]

            node_features.append(type[t])
            index_1.append(int(v))
            index_2.append(int(w))

            # 1 neighbors.
            for n in v.out_neighbors():
                if (n, w) in nodes_to_tuple:
                    s = nodes_to_tuple[(n, w)]
                    e = tuple_graph.add_edge(t, s)

                    matrix_1.append([int(t), int(s)])

            # 2 neighbors.
            for n in w.out_neighbors():
                if (v, n) in nodes_to_tuple:
                    s = nodes_to_tuple[(v, n)]
                    e = tuple_graph.add_edge(t, s)

                    matrix_2.append([int(t), int(s)])

        data_list = []

        data_new = Data()

        edge_index_1 = torch.tensor(matrix_1).t().contiguous()
        edge_index_2 = torch.tensor(matrix_2).t().contiguous()

        data_new.edge_index_1 = edge_index_1
        data_new.edge_index_2 = edge_index_2

        data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)
        data_new.index_1 = torch.from_numpy(np.array(index_1)).to(torch.int64)
        data_new.index_2 = torch.from_numpy(np.array(index_2)).to(torch.int64)

        data_new.y = data.y
        data_new.train_mask = data.train_mask
        data_new.test_mask = data.test_mask

        data_list.append(data_new)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'CORA')
    dataset = Arxiv(path)
    data = dataset[0]


    exit(-1)

    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    print(data.adj_t.size())

    model = SAGE(data.num_features, args.hidden_channels,
                 dataset.num_classes, args.num_layers,
                 args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    #logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            #logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        #logger.print_statistics(run)
    #logger.print_statistics()


if __name__ == "__main__":
    main()