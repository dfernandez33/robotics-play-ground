from scipy.stats import gamma
import numpy as np
import torch
import time
import gym
from pynput import keyboard
from pynput.keyboard import KeyCode
import random
from rl_models import PolicyNetwork
from tamer_model import RewardNetwork

HUMAN_REWARD_SIGNAL = 0.0
IS_HUMAN_TALKING = False
TERMINATE = False


def train(
    env,
    reward_network: RewardNetwork,
    loss_criterion,
    optimizer,
    epochs: int,
    max_length: int,
    num_actions: int,
    window_size: int = 100,
    minibatch_size: int = 10,
):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE
    reward_buffer = []
    window = []
    
    for epoch in range(0, epochs):
        TERMINATE = False
        print(f"Starting Epoch: {epoch}.")
        step_counter = 0
        HUMAN_REWARD_SIGNAL = 0.0
        epoch_reward = 0
        
        while step_counter < max_length:
            if TERMINATE:
                print("This epoch has been aborted.")
                break

            if step_counter == 0:
                state = env.reset()
                state = torch.from_numpy(state.astype(np.float32))
            env.render()
            reward_predictions = reward_network(state)
            best_action = torch.argmax(reward_predictions)

            if random.random() > 0.95:
                best_action = torch.tensor(random.randint(0, 1))

            window.append((state, best_action))

            if HUMAN_REWARD_SIGNAL != 0.0:
                reward_buffer.append(
                    (window[-window_size:], HUMAN_REWARD_SIGNAL, 1 / window_size)
                )
                update_weights(
                    [reward_buffer[-1]], loss_criterion, optimizer, reward_network
                )
                window = []
                HUMAN_REWARD_SIGNAL = 0.0
            else:
                if len(reward_buffer) > minibatch_size:
                    window_sample = random.choices(reward_buffer, k=minibatch_size)
                else:
                    window_sample = reward_buffer
                if window_sample:
                    update_weights(
                        window_sample, loss_criterion, optimizer, reward_network
                    )

            state, reward, TERMINATE, _ = env.step(best_action.item())
            epoch_reward += reward
            state = torch.from_numpy(state.astype(np.float32))
            step_counter += 1
            time.sleep(0.05)

        print(f"Accumulated_reward over epoch {epoch}: {epoch_reward}")

    return reward_network


def update_weights(
    window_sample, loss_criterion, optimizer, reward_network, art_states=20
):
    optimizer.zero_grad()
    total_loss = torch.zeros((1,))
    for sample, human_reward, credit in window_sample:
        for state, action in sample:
            for _ in range(art_states):
                state = generate_artificial_state(state)
                reward_predictions = reward_network(state)
                target = reward_predictions.clone()
                curr_reward = target[action]
                mask = target == curr_reward
                reward_signal = torch.ones_like(target) * human_reward
                target = torch.where(
                    mask, reward_signal, torch.zeros_like(reward_signal)
                )
                total_loss += loss_criterion(reward_predictions, target) * credit
    total_loss.backward()
    optimizer.step()


def generate_artificial_state(state, scale=0.01):
    new_state = []
    for element in state:
        new_state.append(np.random.normal(element, np.abs(element) * scale))
    return torch.tensor(new_state)


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
    trained_agent.eval()
    reward_total = 0
    for _ in range(10):
        reward_epoch = 0
        state = env.reset()
        for i in range(100):
            env.render()
            action = trained_agent(torch.from_numpy(state).float()).argmax(dim=0)
            state, reward, done, _ = env.step(action.item())
            reward_epoch += reward
            if done:
                print("Resetting state!")
                print(f"Trial reward:{reward_epoch}")
                reward_total += reward_epoch
                break
        time.sleep(0.2)
    print(f"Average Reward: {reward_total/10}")


if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    nb_actions = environment.action_space.n
    nb_states = environment.observation_space.shape[0]
    hidden_state = 128
    reward_estimator = RewardNetwork(nb_states, hidden_state, nb_actions)

    optim = torch.optim.AdamW(lr=0.001, params=reward_estimator.parameters())
    loss = torch.nn.MSELoss()

    keyboard_listener = keyboard.Listener(on_press=reward_input_handler,)
    keyboard_listener.start()

    print("Training Agent")
    train(
        environment, reward_estimator, loss, optim, 100, 250, nb_actions, 
    )
    print("Running Verification")
    verify(reward_estimator, environment)