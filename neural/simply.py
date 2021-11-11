import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid


class SimpleLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')

        self.w_1 = torch.nn.Linear(in_channels, out_channels)
        self.w_2 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, features, edge_index):

        features_new =  self.w_2(features)
        feature_self = self.w_1(features)

        out = feature_self + self.propagate(edge_index, x=features_new)

        return out




dataset = Planetoid(root='/tmp/Cora', name='Cora')

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
