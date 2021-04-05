from rl_models import ActorNetwork, CriticNetwork
import torch
from replay_buffer import Memory


class DDPG:
    def __init__(self, nb_inputs: int, actor_outputs: int, hidden_size: int, soft_update_weight: float,
                 discount_factor: float, actor_lr=1e-3, critic_lr=1e-3, max_memory=50000, save_path=None,
                 checkpoint_path=None, cuda=False):
        self.actor_network = ActorNetwork(nb_inputs, hidden_size, actor_outputs)
        self.actor_target = ActorNetwork(nb_inputs, hidden_size, actor_outputs)

        self.critic_network = CriticNetwork(nb_inputs + actor_outputs, hidden_size)
        self.critic_target = CriticNetwork(nb_inputs + actor_outputs, hidden_size)

        self.memory = Memory(max_memory)

        # Copy the network parameters onto their respective target networks
        self.__hard_update()

        self.actor_optim = torch.optim.AdamW(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.AdamW(self.critic_network.parameters(), lr=critic_lr)

        # TODO: Figure out why agent isn't working with CUDA
        self.cuda = cuda
        if self.cuda and torch.cuda.is_available():
            self.__to_cuda()

        self.save_path = save_path
        if checkpoint_path is not None:
            self.__load_from_checkpoint(checkpoint_path)

        self.soft_update_weight = soft_update_weight
        self.discount_factor = discount_factor
        self.criterion = torch.nn.MSELoss()

    def get_action(self, state: torch.Tensor):
        if self.cuda:
            state = state.cuda()
        return self.actor_network(state)

    def update(self, batch_size: int):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size, cuda=self.cuda)

        self.__update_actor(states)
        self.__update_critic(states, actions, next_states, rewards)
        self.__target_update()

    def save_agent(self):
        torch.save(
            {
                "actor_network": self.actor_network.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "actor_optimizer": self.actor_optim.state_dict(),
                "critic_network": self.critic_network.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "critic_optimizer": self.critic_optim.state_dict(),
                "memory": self.memory,
            },
            self.save_path
        )

    def __update_actor(self, states: torch.Tensor):
        policy_loss = -self.critic_network(states, self.actor_network(states)).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

    def __update_critic(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor):
        q_vals = self.critic_network(states, actions)
        next_actions = self.actor_target(next_states)
        next_q = self.critic_target(next_states, next_actions)
        q_prime = rewards + self.discount_factor * next_q

        critic_loss = self.criterion(q_vals, q_prime)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def __target_update(self):
        with torch.no_grad():
            # Soft update the weights of the actor target network
            for target_param, param in zip(self.actor_target.parameters(), self.actor_network.parameters()):
                target_param.copy_(
                    target_param * (1.0 - self.soft_update_weight) + param * self.soft_update_weight
                )

            # Soft update the weights of the critic target network
            for target_param, param in zip(self.critic_target.parameters(), self.critic_network.parameters()):
                target_param.copy_(
                    target_param * (1.0 - self.soft_update_weight) + param * self.soft_update_weight
                )

    def __hard_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic_network.parameters()):
            target_param.data.copy_(param.data)

    def __to_cuda(self):
        self.actor_network.cuda()
        self.actor_target.cuda()
        self.critic_network.cuda()
        self.critic_target.cuda()

    def __load_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.actor_network.load_state_dict(checkpoint["actor_network"])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optim.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_network.load_state_dict(checkpoint["critic_network"])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer"])
        self.memory = checkpoint["memory"]
        print("Loaded from checkpoint: {}".format(checkpoint_path))
