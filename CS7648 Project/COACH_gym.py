import numpy as np
import torch
import time
import gym
from pynput import keyboard
from pynput.keyboard import KeyCode
import random
from rl_models import PolicyNetwork

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
    eligibility_decay: float,
    beta: float,
):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE

    window = []
    eligibility_replay_buffer = {}
    feedback_k = 0

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
            print(f"Eligibility replay buffer len: {len(eligibility_replay_buffer)}")
            if TERMINATE:
                print("This epoch has been aborted.")
                break
            env.render()
            if IS_HUMAN_TALKING:
                time.sleep(7)

            if step_counter:
                window.append((state, best_action, p_best_action, HUMAN_REWARD_SIGNAL))

            if HUMAN_REWARD_SIGNAL != 0.0:
                eligibility_replay_buffer[feedback_k] = window[-5:]
                feedback_k += 1
                window = []
                HUMAN_REWARD_SIGNAL = 0.0

            step_counter += 1
            action_vector = policy_network(state)
            p_best_action, best_action = torch.max(action_vector, dim=0)
            if random.random() > 0.9:
                print("RAND ACTION")
                best_action = torch.tensor(random.randint(0, 1))
                p_best_action = action_vector[best_action]

            if len(eligibility_replay_buffer) > minibatch_size:
                window_sample = random.choices(list(eligibility_replay_buffer), k=minibatch_size)
            else:
                window_sample = list(eligibility_replay_buffer)
            if len(window_sample) != 0:
                eligibility_trace_bar = 0.0
                samples_to_remove = []
                for index in window_sample:
                    sample = eligibility_replay_buffer[index]
                    eligibility_trace = torch.zeros(1)
                    final_human_reward = sample[-1][3]
                    for state, best_action, p_best_action, _ in sample:
                        prob_action = policy_network(state)[best_action]
                        eligibility_trace = eligibility_decay * eligibility_trace + (
                            prob_action / p_best_action.item()
                        )
                    if eligibility_trace.item() < 1e-5:
                        samples_to_remove.append(index)
                    else:
                        eligibility_trace_bar += final_human_reward * eligibility_trace
                print(f"eligibility traceBAR BEFORE: {eligibility_trace_bar}")
                if len(window_sample) > len(samples_to_remove):
                    eligibility_trace_bar = (eligibility_trace_bar / (len(window_sample) - len(samples_to_remove))) + \
                                        beta * -torch.sum(policy_network(state) * torch.log(policy_network(state)), dim=0)
                    print(f"eligibility traceBAR AFTER: {eligibility_trace_bar}")
                    eligibility_trace_bar.backward()
                    optimizer.step()
                for index in samples_to_remove:
                    eligibility_replay_buffer.pop(index)

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
        state, reward, done, _ = env.step(random.randint(0, 1))
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
    hidden_state = 32
    policy_network = PolicyNetwork(nb_states, hidden_state, nb_actions)

    optim = torch.optim.AdamW(
        lr=0.0001, params=policy_network.parameters()
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
        50,
        100,
        16,
        0.35,
        1.5
    )
    print("Running Verification")
    verify(policy_network, environment)