import torch.nn as nn
import torch


class ActorNetwork(nn.Module):
    def __init__(self, nb_inputs: int, nb_hidden: int, nb_outputs: int):
        super(ActorNetwork, self).__init__()
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

class CriticNetwork(nn.Module):
    def __init__(self, nb_inputs: int, nb_hidden: int):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(nb_inputs, nb_hidden),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(nb_hidden, nb_hidden),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(nb_hidden, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=1)
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


class PolicyNetwork(nn.Module):
    def __init__(self, nb_inputs: int, nb_hidden: int, nb_outputs: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(nb_inputs, nb_hidden),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(nb_hidden, nb_hidden),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(nb_hidden, nb_outputs),
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor):
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        return self.softmax(output)
