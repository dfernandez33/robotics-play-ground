from scipy.stats import gamma
import numpy as np
import pandas as pd
import torch
import time
import gym
from pynput import keyboard
from pynput.keyboard import KeyCode
from collections import Counter
import random
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
    window_size: int = 10,
    minibatch_size: int = 16,
    use_hyperball: bool = False,
    art_states: int = 20,
):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE
    reward_buffer = []
    window = []
    feedback_counter_positive = Counter()
    feedback_counter_negative = Counter()
    reward_counter = Counter()

    for epoch in range(0, epochs):
        TERMINATE = False
        print(f"Starting Epoch: {epoch}.")
        step_counter = 0
        HUMAN_REWARD_SIGNAL = 0.0
        epoch_reward = 0
        feedback_counter_positive[epoch] = 0
        feedback_counter_negative[epoch] = 0
        reward_counter[epoch] = 0

        while True:
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
                best_action = torch.tensor(random.randint(0, num_actions - 1))

            window.append((state, best_action))

            if HUMAN_REWARD_SIGNAL != 0.0:
                reward_buffer.append(
                    (
                        window[-window_size:],
                        HUMAN_REWARD_SIGNAL,
                        1 / len(window[-window_size:]),
                    )
                )
                update_weights(
                    [reward_buffer[-1]],
                    loss_criterion,
                    optimizer,
                    reward_network,
                    use_hyperball,
                    art_states,
                )
                window = []
                if HUMAN_REWARD_SIGNAL > 0.0:
                    feedback_counter_positive[epoch] += 1
                else:
                    feedback_counter_negative[epoch] += 1
                HUMAN_REWARD_SIGNAL = 0.0
            else:
                if len(reward_buffer) > minibatch_size:
                    window_sample = random.choices(reward_buffer, k=minibatch_size)
                else:
                    window_sample = reward_buffer
                if window_sample:
                    update_weights(
                        window_sample,
                        loss_criterion,
                        optimizer,
                        reward_network,
                        use_hyperball,
                        art_states,
                    )

            state, reward, TERMINATE, _ = env.step(best_action.item())
            epoch_reward += reward
            state = torch.from_numpy(state.astype(np.float32))
            step_counter += 1
            time.sleep(0.15)

        reward_counter[epoch] = epoch_reward

    return (
        reward_network,
        feedback_counter_positive,
        feedback_counter_negative,
        reward_counter,
    )


def update_weights(
    window_sample, loss_criterion, optimizer, reward_network, use_hyperball, art_states,
):
    optimizer.zero_grad()
    total_loss = torch.zeros((1,))
    for sample, human_reward, credit in window_sample:
        for state, action in sample:
            reward_predictions = reward_network(state)
            target = reward_predictions.clone()
            curr_reward = target[action]
            mask = target == curr_reward
            reward_signal = torch.ones_like(target) * human_reward
            target = torch.where(mask, reward_signal, torch.zeros_like(reward_signal))
            total_loss += loss_criterion(reward_predictions, target) * credit
            if use_hyperball:
                for _ in range(art_states):
                    art_state = generate_artificial_state(state)
                    reward_predictions = reward_network(art_state)
                    target = reward_predictions.clone()
                    curr_reward = target[action]
                    mask = target == curr_reward
                    reward_signal = torch.ones_like(target) * human_reward
                    target = torch.where(
                        mask, reward_signal, torch.zeros_like(reward_signal)
                    )
                    total_loss += (
                        loss_criterion(reward_predictions, target)
                        * credit
                        * 1
                        / art_states
                    )
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
    if key == KeyCode(char="1"):
        HUMAN_REWARD_SIGNAL = -1.0
    elif key == KeyCode(char="9"):
        HUMAN_REWARD_SIGNAL = 1.0


def verify(trained_agent: RewardNetwork, env: gym.Env):
    trained_agent.eval()
    reward_total = 0
    for _ in range(500):
        reward_epoch = 0
        state = env.reset()
        while True:
            action = trained_agent(torch.from_numpy(state).float()).argmax(dim=0)
            state, reward, done, _ = env.step(action.item())
            reward_epoch += reward
            if done:
                reward_total += reward_epoch
                break
    print(f"Average Reward: {reward_total/500}")
    return reward_total / 500


if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    nb_actions = environment.action_space.n
    nb_states = environment.observation_space.shape[0]
    hidden_state = 32
    loss = torch.nn.MSELoss()

    keyboard_listener = keyboard.Listener(on_press=reward_input_handler,)
    keyboard_listener.start()

    print("Training Agent")
    total_pd = pd.DataFrame()
    total_verification_mean = []
    for i in range(5):
        reward_estimator = RewardNetwork(nb_states, hidden_state, nb_actions)
        optim = torch.optim.AdamW(lr=0.005, params=reward_estimator.parameters())
        print(f"----------------Trial {i} starting--------------")
        (
            reward_estimator,
            feedback_counter_positive,
            feedback_counter_negative,
            reward_counter,
        ) = train(
            environment,
            reward_estimator,
            loss,
            optim,
            15,
            100,
            nb_actions,
            use_hyperball=True,
            art_states=20,
        )

        print("Results:")
        for epoch in feedback_counter_positive:
            print(
                f"Positive feedback in epoch {epoch}: {feedback_counter_positive[epoch]} | Negative feedback in epoch {epoch}: {feedback_counter_negative[epoch]} | Reward in epoch {epoch}: {reward_counter[epoch]}"
            )

        final_pd = pd.merge(
            pd.DataFrame(feedback_counter_positive.items()),
            pd.DataFrame(feedback_counter_negative.items()),
            on=0,
            how="inner",
        )

        final_pd = pd.merge(
            final_pd, pd.DataFrame(reward_counter.items()), on=0, how="inner"
        )

        final_pd.columns = ["epoch", "positive feedback", "negative feedback", "reward"]
        final_pd["trial"] = i
        total_pd = pd.concat([total_pd, final_pd])
        print("Running Verification")
        total_verification_mean.append(verify(reward_estimator, environment))
        time.sleep(1.0)

    total_pd.to_csv("no_hyperball_experiment.csv")
    pd.DataFrame(total_verification_mean).to_csv("no_hyperball_verification.csv")
