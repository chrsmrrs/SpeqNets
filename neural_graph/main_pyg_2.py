import os.path as osp

import numpy as np
import torch
import torch.optim as optim
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from tqdm import tqdm

from gnn import GNN
import torch_geometric.transforms as T
from graph_tool.all import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import (InMemoryDataset, Data)
from gnn import GNN

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
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

        print(len(dataset))

        data_list = []
        for i, data in enumerate(dataset):

            print(i)

            x = data.x[:, :2].cpu().detach().numpy()
            edge_index = data.edge_index.cpu().detach().numpy()
            edge_attr = data.edge_attr[:, :2].cpu().detach().numpy()

            # Create graph for easier processing.
            g = Graph(directed=False)
            num_nodes = x.shape[0]

            node_features = {}
            for i in range(num_nodes):
                v = g.add_vertex()
                node_features[v] = x[i]

            rows = list(edge_index[0])
            cols = list(edge_index[1])
            edge_vectors = {}

            for ind, (i, j) in enumerate(zip(rows, cols)):
                e = g.add_edge(i, j, add_missing=False)
                g.ep.edge_features[e] = data.edge_attr[ind]

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

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
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


def main():
    print("###")
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'mol')
    dataset = Mol(path)

    exit()


    feature = 'simple'

    if feature == 'full':
        pass
    elif feature == 'simple':
        print('using simple feature')
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    evaluator = Evaluator("ogbg-molhiv")

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False,
                              num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False,
                             num_workers=0)

    model = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=5, emb_dim=300,
                drop_ratio=0.5, virtual_node=False).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, 101):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
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
