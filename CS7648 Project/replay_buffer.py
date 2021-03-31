import random
from collections import deque
import torch


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size, cuda=False):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        state_batch = torch.cat(state_batch, dim=0)
        action_batch = torch.cat(action_batch, dim=0)
        reward_batch = torch.cat(reward_batch, dim=0)
        next_state_batch = torch.cat(next_state_batch, dim=0)

        if cuda:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)