import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from graph_tool.all import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.datasets import WebKB, Actor
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
from itertools import product, combinations_with_replacement
import itertools

# Compute atomic type for ordered set of vertices of graph g.
def compute_atomic_type(g, vertices):
    edge_list = []

    # Loop over all pairs of vertices.
    for i, v in enumerate(vertices):
        for j, w in enumerate(vertices):
            # Check if edge or self loop.
            if g.edge(v, w):
                edge_list.append((i, j, 1))
            elif not g.edge(v, w):
                edge_list.append((i, j, 2))
            elif v == w:
                edge_list.append((i, j, 3))

    edge_list.sort()
    return hash(tuple(edge_list))


class PPI_2_1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(PPI_2_1, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "tetxffffffreetfrfttas"

    @property
    def processed_file_names(self):
        return "PPtI_2_t1ertffffferetffffdgs"

    def download(self):
        pass

    def process(self):

        dataset = 'texas'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = WebKB(path, dataset)
        data = dataset[0]

        k = 3
        s = 2

        x = data.x.cpu().detach().numpy()
        edge_index = data.edge_index.cpu().detach().numpy()

        # Create graph for easier processing.
        g = Graph(directed=False)
        num_nodes = x.shape[0]

        node_features = {}
        for i in range(num_nodes):
            v = g.add_vertex()
            node_features[v] = x[i]

        rows = list(edge_index[0])
        cols = list(edge_index[1])

        for ind, (i, j) in enumerate(zip(rows, cols)):
            g.add_edge(i, j, add_missing=False)

        # Manage atomic types.
        atomic_type = {}
        atomic_counter = 0

        # True if connected multi-set has been found already.
        multiset_exists = {}

        # List of s-multisets.
        k_multisets = combinations_with_replacement(g.vertices(), r=s)

        # Generate (k,s)-multisets.
        for _ in range(s, k):
            # Contains (k,s)-multisets extended by one vertex.
            ext_multisets = []
            # Iterate over every multiset.
            for ms in k_multisets:
                # Iterate over every element in multiset.
                for v in ms:
                    # Iterate over neighbors.
                    for w in v.out_neighbors():
                        # Extend multiset by neighbor w.
                        new_multiset = list(ms[:])
                        new_multiset.append(w)
                        new_multiset.sort()

                        # Check if set already exits to avoid duplicates.
                        if not (tuple(new_multiset) in multiset_exists):
                            multiset_exists[tuple(new_multiset)] = True
                            ext_multisets.append(list(new_multiset[:]))

                    # Self-loop.
                    new_multiset = list(ms[:])
                    new_multiset.append(v)
                    new_multiset.sort()

                    # Check if set already exits to avoid duplicates.
                    if not (tuple(new_multiset) in multiset_exists):
                        multiset_exists[tuple(new_multiset)] = True
                        ext_multisets.append(new_multiset)

            k_multisets = ext_multisets

        if k == s:
            k_multisets = list(k_multisets)

        # True if tuple exists in k-tuple graph.
        tuple_exists = {}

        tuple_graph = Graph(directed=False)
        type = {}

        tuple_to_nodes = {}
        nodes_to_tuple = {}

        # Generate nodes of (k,s)-graph.
        # Iterate of (k,s)-multisets.
        for ms in k_multisets:
            # Create all permutations of multiset.
            permutations = itertools.permutations(ms)

            # Iterate over permutations of multiset.
            for t in permutations:
                # Check if tuple t already exists. # TODO: Needed?
                if t not in tuple_exists:
                    tuple_exists[t] = True

                    # Add vertex to k-tuple graph representing tuple t.
                    t_v = tuple_graph.add_vertex()


                    # Compute atomic type.
                    raw_type = compute_atomic_type(g, t)

                    at = 0
                    # Atomic type seen before.
                    if raw_type in atomic_type:
                        at = atomic_type[raw_type]
                    else:  # Atomic type not seen before.
                        at = atomic_counter
                        atomic_type[raw_type] = atomic_counter
                        atomic_counter += 1

                    tmp = np.concatenate([node_features[i] for i in t], axis=-1)

                    one_hot = np.zeros((80,))

                    one_hot[int(at)] = 1
                    type[t_v] = np.concatenate([one_hot,tmp])


                    #type[t_v] = tmp


                    # Manage mappings, back and forth.
                    tuple_to_nodes[t_v] = t
                    nodes_to_tuple[t] = t_v

        matrix_1 = []
        matrix_2 = []
        matrix_3 = []
        node_features = []

        index_1 = []
        index_2 = []
        index_3 = []

        for c, t in enumerate(tuple_graph.vertices()):

            print(c, tuple_graph.num_vertices())
            # Get underlying nodes.
            v, w, u = tuple_to_nodes[t]

            node_features.append(type[t])
            index_1.append(int(v))
            index_2.append(int(w))
            index_3.append(int(u))


            # 1 neighbors.
            for n in v.out_neighbors():
                if (n, w, u) in nodes_to_tuple:
                    s = nodes_to_tuple[(n, w, u)]
                    matrix_1.append([int(t), int(s)])

            # 2 neighbors.
            for n in w.out_neighbors():
                if (v, n, u) in nodes_to_tuple:
                    s = nodes_to_tuple[(v, n, u)]
                    matrix_2.append([int(t), int(s)])

            # 3 neighbors.
            for n in w.out_neighbors():
                if (v, w, n) in nodes_to_tuple:
                    s = nodes_to_tuple[(v, w, n)]
                    matrix_3.append([int(t), int(s)])

        data_list = []

        data_new = Data()

        edge_index_1 = torch.tensor(matrix_1).t().contiguous()
        edge_index_2 = torch.tensor(matrix_2).t().contiguous()
        edge_index_3 = torch.tensor(matrix_3).t().contiguous()

        data_new.edge_index_1 = edge_index_1
        data_new.edge_index_2 = edge_index_2
        data_new.edge_index_3 = edge_index_3

        data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)
        data_new.index_1 = torch.from_numpy(np.array(index_1)).to(torch.int64)
        data_new.index_2 = torch.from_numpy(np.array(index_2)).to(torch.int64)
        data_new.index_3 = torch.from_numpy(np.array(index_3)).to(torch.int64)

        data_new.y = data.y

        data_new.train_mask = data.train_mask
        data_new.test_mask = data.test_mask
        data_new.val_mask = data.val_mask

        data_list.append(data_new)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in [
            'edge_index_1', 'edge_index_2', 'edge_index_3', 'index_1', 'index_2',  'index_3'

        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ttertee')
dataset = PPI_2_1(path, transform=MyTransform())
data = dataset[0]



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        dim = 256
        self.conv_1_1 = GCNConv(5189, dim)
        self.conv_1_2 = GCNConv(5189, dim)
        self.conv_1_3 = GCNConv(5189, dim)

        self.mlp_1 = Sequential(Linear(3 * dim, dim), ReLU(), Linear(dim, dim))

        self.conv_2_1 = GCNConv(dim, dim)
        self.conv_2_2 = GCNConv(dim, dim)
        self.conv_2_3 = GCNConv(dim, dim)
        self.mlp_2 = Sequential(Linear(3 * dim, dim), ReLU(), Linear(dim, dim))

        self.mlp = Sequential(Linear(3 * dim, dim), ReLU(), Linear(dim, 7))

    def forward(self):
        x, edge_index_1, edge_index_2, edge_index_3 = data.x, data.edge_index_1, data.edge_index_2, data.edge_index_3

        index_1, index_2, index_3 = data.index_1, data.index_2, data.index_3

        x_1 = F.relu(self.conv_1_1(x, edge_index_1))
        x_2 = F.relu(self.conv_1_2(x, edge_index_2))
        x_3 = F.relu(self.conv_1_3(x, edge_index_3))
        x = self.mlp_1(torch.cat([x_1, x_2, x_3], dim=-1))


        x_1 = F.relu(self.conv_2_1(x, edge_index_1))
        x_2 = F.relu(self.conv_2_2(x, edge_index_2))
        x_3 = F.relu(self.conv_2_3(x, edge_index_3))
        x = self.mlp_2(torch.cat([x_1, x_2, x_3], dim=-1))

        index_1 = index_1.to(torch.int64)
        index_2 = index_2.to(torch.int64)
        index_2 = index_3.to(torch.int64)
        x_1 = scatter(x, index_1, dim=0, reduce="mean")
        x_2 = scatter(x, index_2, dim=0, reduce="mean")
        x_3 = scatter(x, index_3, dim=0, reduce="mean")
        x = self.mlp(torch.cat([x_1, x_2, x_3], dim=1))

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


for i in range(5):
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