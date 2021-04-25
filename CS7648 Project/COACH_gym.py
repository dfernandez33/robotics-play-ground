from scipy.stats import gamma
import numpy as np
import torch
import time
import argparse
import gym
from pynput import keyboard, mouse
from pynput.keyboard import KeyCode
from pynput.mouse import Button
import random
from utils import calculate_reward
import pandas as pd
from rl_models import PolicyNetwork

from language_model.model import BertTransformerVerbalReward

HUMAN_REWARD_SIGNAL = 0.0
IS_HUMAN_TALKING = False
TERMINATE = False


def train(
    env,
    policy_network: PolicyNetwork,
    loss_criterion,
    optimizer,
    starting_epoch: int,
    epochs: int,
    max_length: int,
    minibatch_size: int,
    eligibility_decay: int,
    beta: int,
):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE

    window = []
    eligibility_replay_buffer = []

    for epoch in range(starting_epoch, starting_epoch + epochs + 1):
        TERMINATE = False
        print("Resetting Robot Arm")
        state = env.reset()
        print(f"Starting Epoch: {epoch}.")
        step_counter = 0
        HUMAN_REWARD_SIGNAL = 0.0
        while step_counter < max_length:
            env.render()
            if IS_HUMAN_TALKING:
                time.sleep(7)

            if step_counter:
                window.append((state, best_action, p_best_action, HUMAN_REWARD_SIGNAL))

            if HUMAN_REWARD_SIGNAL != 0.0:
                eligibility_replay_buffer.append(window)
                window = []
                HUMAN_REWARD_SIGNAL = 0.0

            step_counter += 1
            action_vector = policy_network(state)
            p_best_action, best_action = torch.max(action_vector, dim=0)
            if random.random() > 0.95:
                print("RAND ACTION")
                best_action = random.randint(0, 7)
                p_best_action = action_vector[best_action]

            if len(eligibility_replay_buffer) > minibatch_size:
                window_sample = random.sample(eligibility_replay_buffer, minibatch_size)
            else:
                window_sample = eligibility_replay_buffer

            eligibility_trace_bar = 0
            for sample in window_sample:
                eligibility_trace = 0
                final_human_reward = sample[-1][3]
                for state, best_action, p_best_action, _ in sample:
                    prob_action = policy_network(state)[best_action]
                    eligibility_trace = eligibility_decay * eligibility_trace + (
                        prob_action / p_best_action
                    ) * torch.log(prob_action)
                eligibility_trace_bar += final_human_reward * eligibility_trace

            eligibility_trace_bar = (
                eligibility_trace_bar / minibatch_size
            ) + beta * -torch.sum(
                policy_network(state) * torch.log(policy_network(state)), axis=0
            )
            eligibility_trace_bar.backward()
            optimizer.step()

            if TERMINATE:
                print("This epoch has been aborted.")
                break

            state, reward, TERMINATE, _ = env.step(best_action)

    return policy_network


def reward_input_handler(key):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE
    if key == KeyCode(char="q"):
        TERMINATE = True
        HUMAN_REWARD_SIGNAL = -50
    elif key == KeyCode(char="w"):
        TERMINATE = True
        HUMAN_REWARD_SIGNAL = 50
    elif key == KeyCode(char="1"):
        HUMAN_REWARD_SIGNAL = -2.0
    elif key == KeyCode(char="2"):
        HUMAN_REWARD_SIGNAL = -1.5
    elif key == KeyCode(char="3"):
        HUMAN_REWARD_SIGNAL = -1.0
    elif key == KeyCode(char="4"):
        HUMAN_REWARD_SIGNAL = -0.5
    elif key == KeyCode(char="5"):
        HUMAN_REWARD_SIGNAL = 0.0
    elif key == KeyCode(char="6"):
        HUMAN_REWARD_SIGNAL = 0.5
    elif key == KeyCode(char="7"):
        HUMAN_REWARD_SIGNAL = 1.0
    elif key == KeyCode(char="8"):
        HUMAN_REWARD_SIGNAL = 1.5
    elif key == KeyCode(char="9"):
        HUMAN_REWARD_SIGNAL = 2.0


def verify(trained_agent: PolicyNetwork, env: gym.Env):
    state = env.reset()
    trained_agent.actor_network.eval()
    for _ in range(500):
        env.render()
        action = trained_agent.get_action(torch.from_numpy(state).float())
        state, reward, done, _ = env.step(action.detach().numpy())

if __name__ == '__main__':
    environment = gym.make('Pendulum-v0')
    nb_actions = environment.action_space.shape[0]
    nb_states = environment.observation_space.shape[0]
    hidden_state = 128
    policy_network = PolicyNetwork(nb_states, hidden_state, nb_actions)

    optim = torch.optim.AdamW(
        lr=0.00025, params=policy_network.parameters()
    )
    loss = torch.nn.MSELoss()

    keyboard_listener = keyboard.Listener(on_press=reward_input_handler,)
    keyboard_listener.start()

    print("Training Agent")
    train(
        environment,
        policy_network,
        loss,
        optim,
        0,
        25,
        50,
        16,
        0.35,
        0.15
    )
    print("Running Verification")
    verify(policy_network, environment)