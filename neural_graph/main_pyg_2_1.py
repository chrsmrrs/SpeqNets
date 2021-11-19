import os.path as osp

import numpy as np
import torch
import torch.optim as optim
from graph_tool.all import *
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import torch.nn.functional as F

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


class Mol(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(Mol, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "Mol"

    @property
    def processed_file_names(self):
        return "Mol"

    def download(self):
        pass

    def process(self):
        dataset = PygGraphPropPredDataset(name="ogbg-molbbbp")

        print(len(dataset))
        atom_encoder = AtomEncoder(100)
        bond_encoder = BondEncoder(100)

        data_list = []
        for i, data in enumerate(dataset):

            print(i)

            # x = atom_encoder(data.x[:, :2]).cpu().detach().numpy()
            # edge_attr = bond_encoder(data.edge_attr[:, :2]).cpu().detach().numpy()

            x = atom_encoder(data.x).cpu().detach().numpy()
            edge_attr = bond_encoder(data.edge_attr).cpu().detach().numpy()

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
            edge_features = {}

            for ind, (i, j) in enumerate(zip(rows, cols)):
                e = g.add_edge(i, j, add_missing=False)
                edge_features[e] = edge_attr[ind]

            type = {}

            tuple_to_nodes = {}
            nodes_to_tuple = {}

            tuples = 0
            for v in g.vertices():
                for w in v.all_neighbors():
                    n = tuples
                    tuples += 1

                    tuple_to_nodes[n] = (v, w)
                    nodes_to_tuple[(v, w)] = n

                    type[n] = np.concatenate(
                        [node_features[v], node_features[w], edge_features[g.edge(v, w)], np.array([1, 0])], axis=-1)

                n = tuples
                tuples += 1

                tuple_to_nodes[n] = (v, v)
                tuple_to_nodes[(v, v)] = n
                type[n] = np.concatenate([node_features[v], node_features[v], [0.0] * 100, np.array([0, 1])], axis=-1)

            matrix_1 = []
            matrix_2 = []
            node_features_t = []

            for t in range(tuples):
                v, w = tuple_to_nodes[t]

                node_features_t.append(type[t])

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

            data_new = Data()

            data_new.edge_index_1 = torch.tensor(matrix_1).t().contiguous().to(torch.long)
            data_new.edge_index_2 = torch.tensor(matrix_2).t().contiguous().to(torch.long)

            data_new.x = torch.from_numpy(np.array(node_features_t)).to(torch.float)
            data_new.y = data.y

            data_list.append(data_new)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))

        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GNN(torch.nn.Module):
    def __init__(self, num_tasks):
        super(GNN, self).__init__()

        dim = 302

        self.conv_1_1 = GINConv(dim)
        self.conv_1_2 = GINConv(dim)
        self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))
        self.bn_1 = torch.nn.BatchNorm1d(dim)

        self.conv_2_1 = GINConv(dim)
        self.conv_2_2 = GINConv(dim)
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))
        self.bn_2 = torch.nn.BatchNorm1d(dim)

        self.conv_3_1 = GINConv(dim)
        self.conv_3_2 = GINConv(dim)
        self.mlp_3 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))
        self.bn_3 = torch.nn.BatchNorm1d(dim)

        self.conv_4_1 = GINConv(dim)
        self.conv_4_2 = GINConv(dim)
        self.mlp_4 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))
        self.bn_4 = torch.nn.BatchNorm1d(dim)

        self.conv_5_1 = GINConv(dim)
        self.conv_5_2 = GINConv(dim)
        self.mlp_5 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))
        self.bn_5 = torch.nn.BatchNorm1d(dim)

        self.pool = global_mean_pool
        self.graph_pred_linear = torch.nn.Linear(1*dim, num_tasks)

    def forward(self, batched_data):
        x, edge_index_1, edge_index_2, batch = batched_data.x, batched_data.edge_index_1, batched_data.edge_index_2, batched_data.batch

        x_out = x

        x_1 = self.conv_1_1(x, edge_index_1)
        x_2 = self.conv_1_2(x, edge_index_2)
        x = self.mlp_1(torch.cat([x_1, x_2], dim=-1))
        x_1_out = self.bn_1(x)
        x_1_out = F.dropout(F.relu(x_1_out), 0.5, training=self.training)

        x_1 = self.conv_2_1(x_1_out, edge_index_1)
        x_2 = self.conv_2_2(x_1_out, edge_index_2)
        x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))
        x_2_out = self.bn_1(x)
        x_2_out = F.dropout(F.relu(x_2_out), 0.5, training=self.training)

        x_1 = self.conv_3_1(x_2_out, edge_index_1)
        x_2 = self.conv_3_2(x_2_out, edge_index_2)
        x = self.mlp_3(torch.cat([x_1, x_2], dim=-1))
        x_3_out = self.bn_1(x)
        x_3_out = F.dropout(F.relu(x_3_out), 0.5, training=self.training)

        x_1 = self.conv_4_1(x_3_out, edge_index_1)
        x_2 = self.conv_4_2(x_3_out, edge_index_2)
        x = self.mlp_4(torch.cat([x_1, x_2], dim=-1))
        x_4_out = self.bn_1(x)
        x_4_out = F.dropout(F.relu(x_4_out), 0.5, training=self.training)

        x_1 = self.conv_5_1(x_4_out, edge_index_1)
        x_2 = self.conv_5_2(x_4_out, edge_index_2)
        x = self.mlp_5(torch.cat([x_1, x_2], dim=-1))
        x_5_out = self.bn_1(x)
        x_5_out = F.dropout(x_5_out, 0.5, training=self.training)

        x = self.pool(x_5_out, batched_data.batch)

        return self.graph_pred_linear(x)


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


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


def main():
    dataset_base = PygGraphPropPredDataset(name="ogbg-molbbbp")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'mol')
    dataset = Mol(path, transform=MyTransform())

    print(dataset.data.x.size())

    split_idx = dataset_base.get_idx_split()

    evaluator = Evaluator("ogbg-molbbbp")

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False,
                              num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False,
                             num_workers=0)

    model = GNN(dataset_base.num_tasks).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, 101):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset_base.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset_base.eval_metric])
        valid_curve.append(valid_perf[dataset_base.eval_metric])
        test_curve.append(test_perf[dataset_base.eval_metric])

    if 'classification' in dataset_base.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))


if __name__ == "__main__":
    main()


# ogbg-moltoxcast: 0.7320998196137835
# ogbg-molbbbp: 0.6440007716049383
ogbg-mollipo: