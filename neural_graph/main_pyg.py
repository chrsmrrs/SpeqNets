import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN(torch.nn.Module):
    def __init__(self, num_tasks, emb_dim):
        super(GNN, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim)

        self.conv_1 = GINConv(emb_dim)
        self.bn_1 = torch.nn.BatchNorm1d(emb_dim)
        self.conv_2 = GINConv(emb_dim)
        self.bn_2 = torch.nn.BatchNorm1d(emb_dim)
        self.conv_3 = GINConv(emb_dim)
        self.bn_3 = torch.nn.BatchNorm1d(emb_dim)
        self.conv_4 = GINConv(emb_dim)
        self.bn_4 = torch.nn.BatchNorm1d(emb_dim)
        self.conv_5 = GINConv(emb_dim)
        self.bn_5 = torch.nn.BatchNorm1d(emb_dim)

        self.pool = global_mean_pool
        self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x = self.atom_encoder(x)

        x = self.conv_1(x, edge_index, edge_attr)
        x = self.bn_1(x)
        x = F.dropout(F.relu(x), 0.5, training=self.training)

        x = self.conv_2(x, edge_index, edge_attr)
        x = self.bn_2(x)
        x = F.dropout(F.relu(x), 0.5, training=self.training)

        x = self.conv_3(x, edge_index, edge_attr)
        x = self.bn_3(x)
        x = F.dropout(F.relu(x), 0.5, training=self.training)

        x = self.conv_4(x, edge_index, edge_attr)
        x = self.bn_4(x)
        x = F.dropout(F.relu(x), 0.5, training=self.training)

        x = self.conv_5(x, edge_index, edge_attr)
        x = self.bn_5(x)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.pool(x, batched_data.batch)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PygGraphPropPredDataset(name="ogbg-moltoxcast")
    print(len(dataset))

    feature = 'full'

    if feature == 'full':
        pass
    elif feature == 'simple':
        print('using simple feature')
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    evaluator = Evaluator("ogbg-molsider")

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False,
                              num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False,
                             num_workers=0)

    model = GNN(dataset.num_tasks, 100).to(device)

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

# ogbg-moltoxcast
# ogbg-molsider