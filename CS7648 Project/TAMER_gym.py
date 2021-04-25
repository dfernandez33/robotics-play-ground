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
from tamer_model import RewardNetwork
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
        state = torch.from_numpy(state.astype(np.float32))
        print(f"Starting Epoch: {epoch}.")
        step_counter = 0
        HUMAN_REWARD_SIGNAL = 0.0
        epoch_reward = 0
        while step_counter < max_length:
            env.render()
            if IS_HUMAN_TALKING:
                time.sleep(7)

            if step_counter:
                window.append((state, best_action, p_best_action, HUMAN_REWARD_SIGNAL))

            if HUMAN_REWARD_SIGNAL != 0.0:
                eligibility_replay_buffer.append(window[-20:])
                window = []
                HUMAN_REWARD_SIGNAL = 0.0

            step_counter += 1
            action_vector = policy_network(state)
            p_best_action, best_action = torch.max(action_vector, dim=0)
            if random.random() > 0.95:
                print("RAND ACTION")
                best_action = torch.tensor(random.randint(0, 1))
                p_best_action = action_vector[best_action]

            if len(eligibility_replay_buffer) > minibatch_size:
                window_sample = random.sample(eligibility_replay_buffer, minibatch_size)
            else:
                window_sample = eligibility_replay_buffer

            eligibility_trace_bar = 0.0
            for sample in window_sample:
                eligibility_trace = torch.zeros(1)
                final_human_reward = sample[-1][3]
                for state, best_action, p_best_action, _ in sample:
                    prob_action = policy_network(state)[best_action]
                    eligibility_trace = eligibility_decay * eligibility_trace + (
                        prob_action / p_best_action.item()
                    ) * torch.log(prob_action)

                eligibility_trace_bar += final_human_reward * eligibility_trace
            print(eligibility_trace_bar)
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

            state, reward, TERMINATE, _ = env.step(best_action.item())
            epoch_reward += reward
            state = torch.from_numpy(state.astype(np.float32))
            time.sleep(0.5)
        print('EPOCH REWARD')
        print(epoch_reward)
        print('MODEL PARAMS')
        for name, param in policy_network.named_parameters():
            print(name)
            print(param.sum())
            print('-----------')


    return policy_network


def train(
    env,
    reward_network: RewardNetwork,
    loss_criterion,
    optimizer,
    starting_epoch: int,
    epochs: int,
    max_length: int,
):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE
    epoch_reward = 0
    for epoch in range(0, epochs):
        TERMINATE = False
        print("Resetting Robot Arm")
        state = env.reset()
        state = torch.from_numpy(state.astype(np.float32))
        print(f"Starting Epoch: {epoch}.")
        step_counter = 0
        HUMAN_REWARD_SIGNAL = 0.0
        creditor = {}
        epoch_reward = 0
        while step_counter < max_length:
            env.render()
            step_counter += 1
            reward_predictions = reward_network(state)
            best_action = torch.argmax(reward_predictions)
            if random.random() > 0.95:
                print("RAND ACTION")
                best_action = torch.tensor(random.randint(0, 1))
            creditor[step_counter] = (state, best_action, time.time())

            if HUMAN_REWARD_SIGNAL != 0.0:
                update_weights(HUMAN_REWARD_SIGNAL, time.time(), creditor, loss_criterion, optimizer, reward_network)
                HUMAN_REWARD_SIGNAL = 0.0
                creditor = {}

            if TERMINATE:
                print("This epoch has been aborted.")
                break

            state, reward, TERMINATE, _ = env.step(best_action.item())
            epoch_reward += reward
            state = torch.from_numpy(state.astype(np.float32))
            time.sleep(0.5)

        print(f'Accumulated_reward over epoch {epoch}: {epoch_reward}')

    return reward_network


def update_weights(reward_signal: float, human_time: float, creditor, loss_criterion, optimizer, reward_network):
    optimizer.zero_grad()
    total_loss = torch.zeros((1,))
    for state, best_action, action_time in creditor.values():
        reward_predictions = reward_network(state)
        print(reward_predictions)
        credit = gamma.pdf((human_time - action_time), 1, 0.0, 0.15)
        target = reward_predictions.clone()
        curr_reward = target[best_action]
        mask = (target == curr_reward)
        reward_signal = torch.ones_like(target) * reward_signal
        target = torch.where(mask, reward_signal, torch.zeros_like(reward_signal))
        print(credit * reward_predictions)
        print(target * credit)
        print('--=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=----')
        total_loss += loss_criterion(credit * reward_predictions, target * credit)
    total_loss.backward()
    optimizer.step()
    print(f'Loss for this update: {total_loss}')

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
    trained_agent.eval()
    reward_total = 0
    for _ in range(500):
        env.render()
        action = trained_agent(torch.from_numpy(state).float()).argmax(dim=0)
        state, reward, done, _ = env.step(action.item())
        reward_total += reward
        if done:
            print('Resetting state!')
            state = env.reset()
            print(reward_total)
            reward_total = 0
        time.sleep(0.2)

if __name__ == '__main__':
    environment = gym.make('CartPole-v0')
    nb_actions = environment.action_space.n
    nb_states = environment.observation_space.shape[0]
    hidden_state = 128
    reward_estimator = RewardNetwork(nb_states, hidden_state, nb_actions)

    optim = torch.optim.AdamW(
        lr=0.001, params=reward_estimator.parameters()
    )
    loss = torch.nn.MSELoss()

    keyboard_listener = keyboard.Listener(on_press=reward_input_handler,)
    keyboard_listener.start()

    print("Training Agent")
    train(
        environment,
        reward_estimator,
        loss,
        optim,
        0,
        50,
        100,
    )
    print("Running Verification")
    verify(reward_estimator, environment)