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
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU


from typing import Union, Optional

from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import sort_edge_index
from torch_geometric.transforms import BaseTransform


class ToSparseTensor(BaseTransform):
    r"""Converts the :obj:`edge_index` attributes of a homogeneous or
    heterogeneous data object into a (transposed)
    :class:`torch_sparse.SparseTensor` type with key :obj:`adj_.t`.

    .. note::

        In case of composing multiple transforms, it is best to convert the
        :obj:`data` object to a :obj:`SparseTensor` as late as possible, since
        there exist some transforms that are only able to operate on
        :obj:`data.edge_index` for now.

    Args:
        attr: (str, optional): The name of the attribute to add as a value to
            the :class:`~torch_sparse.SparseTensor` object (if present).
            (default: :obj:`edge_weight`)
        remove_edge_index (bool, optional): If set to :obj:`False`, the
            :obj:`edge_index` tensor will not be removed.
            (default: :obj:`True`)
        fill_cache (bool, optional): If set to :obj:`False`, will not fill the
            underlying :obj:`SparseTensor` cache. (default: :obj:`True`)
    """
    def __init__(self, attr: Optional[str] = 'edge_weight',
                 remove_edge_index: bool = True, fill_cache: bool = True):
        self.attr = attr
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index_1'  not in store and 'edge_index_2'  not in stor:
                continue

            nnz = store.edge_index.size(1)

            keys, values = [], []
            for key, value in store.items():
                if isinstance(value, Tensor) and value.size(0) == nnz:
                    keys.append(key)
                    values.append(value)

            store.edge_index, values = sort_edge_index(store.edge_index,
                                                       values,
                                                       sort_by_row=False)

            for key, value in zip(keys, values):
                store[key] = value

            store.adj_t = SparseTensor(
                row=store.edge_index[1], col=store.edge_index[0],
                value=None if self.attr is None or self.attr not in store else
                store[self.attr], sparse_sizes=store.size()[::-1],
                is_sorted=True)

            if self.remove_edge_index:
                del store['edge_index']
                if self.attr is not None and self.attr in store:
                    del store[self.attr]

            if self.fill_cache:  # Pre-process some important attributes.
                store.adj_t.storage.rowptr()
                store.adj_t.storage.csr2csc()

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class PPI_2_1(InMemoryDataset):
    def __init__(self, split, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(PPI_2_1, self).__init__( root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "PPI_2_1"

    @property
    def processed_file_names(self):
        return "PPI_2_1"

    def download(self):
        pass

    def process(self):

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "PPI")
        dataset = PPI(path, split=self.split)
        data = dataset[0]

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
                    [node_features[v], node_features[w], np.array([1, 0])], axis=-1)

            n = tuple_graph.add_vertex()
            tuple_to_nodes[n] = (v, v)
            tuple_to_nodes[(v, v)] = n
            type[n] = np.concatenate([node_features[v], node_features[v], np.array([0, 1])], axis=-1)

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
                    matrix_1.append([int(t), int(s)])

            # 2 neighbors.
            for n in w.out_neighbors():
                if (v, n) in nodes_to_tuple:
                    s = nodes_to_tuple[(v, n)]
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

        data_list.append(data_new)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in [
            'edge_index_1', 'edge_index_2', 'index_1', 'index_2'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data

pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'eee')
train_dataset = PPI_2_1("train", path, pre_transform=pre_transform, transform=MyTransform(), )
val_dataset = PPI_2_1("val", path, pre_transform=pre_transform, transform=MyTransform(), )
test_dataset = PPI_2_1("test", path, pre_transform=pre_transform, transform=MyTransform(), )

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 256

        self.conv_1_1 = GINConv(
            Sequential(Linear(1000, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.conv_1_2 = GINConv(
            Sequential(Linear(1000, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.conv_2_1 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.conv_2_1 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.conv_3_1 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.conv_3_2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, 121)


    def forward(self, x, edge_index_1, edge_index_2):

        x_1 = F.relu(self.conv_1_1(x, edge_index_1))
        x_2 = F.relu(self.conv_1_2(x, edge_index_2))
        x = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv_2_1(x, edge_index_1))
        x_2 = F.relu(self.conv_2_2(x, edge_index_2))
        x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv_3_1(x, edge_index_1))
        x_2 = F.relu(self.conv_3_2(x, edge_index_2))
        x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))


        index_1 = index_1.to(torch.int64)
        index_2 = index_2.to(torch.int64)
        x_1 = scatter(x, index_1, dim=0, reduce="mean")
        x_2 = scatter(x, index_2, dim=0, reduce="mean")


        exit()

        x = self.mlp(torch.cat([x_1, x_2], dim=1))

        return F.log_softmax(x, dim=1)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()



def train():
    model.train()

    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data.x, data.adj_t), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.adj_t.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for epoch in range(1, 2001):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))
