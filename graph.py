import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Encoder(torch.nn.Module):
    def __init__(self, num_input, hidden_size):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(8, 12)
        self.conv1 = GCNConv(num_input + 12, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

    def forward(self, node, node_type, edge_index):
        num_graph = node.shape[0]
        node_type = node_type.repeat(num_graph, 1)
        embeddings = self.embed(node_type)
        node = torch.cat([node, embeddings], dim=2)
        x = self.conv1(node, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.mean(x, axis=1)
        return x


class Actor(nn.Module):
    def __init__(self, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 32)
        self.mu_head = nn.Linear(32, 1)
        self.sigma_head = nn.Linear(32, 1)

    def forward(self, s):

        l1 = torch.tanh(self.fc1(s))
        mu = torch.tanh(self.mu_head(l1)) * 0.1
        sigma = torch.tensor([0.05]).expand_as(mu).cuda()

        return mu, sigma


class Critic(nn.Module):
    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(hidden_size, 32)
        self.state_value = nn.Linear(32, 1)

    def forward(self, s):

        l1 = F.leaky_relu(self.fc1(s))
        value = self.state_value(l1)
        return value