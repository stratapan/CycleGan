import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.main(x)


class MLP_Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.main(x)