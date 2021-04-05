import torch.nn as nn
import torch


class RewardNetwork(nn.Module):
    def __init__(self, nb_inputs: int, nb_hidden: int, nb_outputs: int):
        super(RewardNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(nb_inputs, nb_hidden),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(nb_hidden, nb_hidden),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(nb_hidden, nb_outputs)
        )

    def forward(self, x: torch.Tensor):
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
