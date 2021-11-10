import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')

class SimpleLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')

        self.w_1 = torch.nn.Linear(in_channels, out_channels)
        self.w_2 = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):

        x =  self.w_1(x)
        f_new = self.w_2(x)

        out = f_new + self.propagate(edge_index, x=x)

        return out


class SimpleArchitecture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SimpleLayer(dataset.num_node_features, 16)
        self.conv2 = SimpleLayer(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleArchitecture().to(device)
data = dataset[0].to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
