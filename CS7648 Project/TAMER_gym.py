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
import speech_recognition as sr
from language_model.model import BertTransformerVerbalReward
import math


HUMAN_REWARD_SIGNAL = 0.0
IS_HUMAN_TALKING = False
TERMINATE = False
SOFT_UPDATE_WEIGHT = .01
TRUST_DECAY_START = 0.9
TRUST_DECAY_END = 0.05
TRUST_DECAY_RATE = 100
STEPS_DONE = 0
device = torch.device("cpu")


def train(
    env,
    reward_network: RewardNetwork,
    target_reward_network: RewardNetwork,
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
    global STEPS_DONE
    global device
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
            if not epoch:
                time.sleep(0.15)
            STEPS_DONE += 1
            if TERMINATE:
                print("This epoch has been aborted.")
                break

            if IS_HUMAN_TALKING:
                time.sleep(5.5)

            if step_counter == 0:
                state = env.reset()
                state = torch.from_numpy(state.astype(np.float32)).to(device)
            env.render()
            if len(reward_buffer) == 0:
                time.sleep(0.15)
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
                    target_network=target_reward_network
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
                        target_network=target_reward_network
                    )

            state, reward, TERMINATE, _ = env.step(best_action.item())
            epoch_reward += reward
            state = torch.from_numpy(state.astype(np.float32)).to(device)
            step_counter += 1
            print('...............')

        reward_counter[epoch] = epoch_reward

    return (
        reward_network,
        feedback_counter_positive,
        feedback_counter_negative,
        reward_counter,
    )


def update_weights(
    window_sample, loss_criterion, optimizer, reward_network, use_hyperball, art_states, target_network=None
):
    global STEPS_DONE
    global device
    optimizer.zero_grad()
    total_loss = torch.zeros((1,)).to(device)
    for sample, human_reward, credit in window_sample:
        for state, action in sample:
            state = state.to(device)
            reward_predictions = reward_network(state)
            target = reward_predictions.clone()
            curr_reward = target[action]
            mask = target == curr_reward
            reward_signal = torch.ones_like(target) * human_reward
            target = torch.where(mask, reward_signal, torch.zeros_like(reward_signal))
            total_loss += loss_criterion(reward_predictions, target) * credit
            if use_hyperball:
                for _ in range(art_states):
                    art_state = generate_artificial_state(state).to(device)
                    reward_predictions = reward_network(art_state)
                    target = reward_predictions.clone()
                    curr_reward = target[action]
                    mask = target == curr_reward
                    trust_weight = TRUST_DECAY_END + (TRUST_DECAY_START - TRUST_DECAY_END) \
                                   * math.exp(-1 * STEPS_DONE / TRUST_DECAY_RATE)
                    reward_signal = trust_weight * (torch.ones_like(target) * human_reward) \
                                    + (1 - trust_weight) * target_reward_estimator(state)
                    target = torch.where(
                        mask, reward_signal, torch.zeros_like(reward_signal)
                    )
                    total_loss += (
                        loss_criterion(reward_predictions, target)
                        * credit
                    )
    total_loss.backward()
    optimizer.step()
    if use_hyperball:
        target_update(reward_network, target_network)


def generate_artificial_state(state, scale=0.01):
    new_state = []
    for element in state:
        if torch.cuda.is_available():
            new_state.append(np.random.normal(element.item(), np.abs(element.item()) * scale))
        else:
            new_state.append(np.random.normal(element, np.abs(element) * scale))
    return torch.tensor(new_state)


def reward_input_handler(key):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE
    global IS_HUMAN_TALKING
    if key == KeyCode(char="1"):
        HUMAN_REWARD_SIGNAL = -1.0
    elif key == KeyCode(char="9"):
        HUMAN_REWARD_SIGNAL = 1.0
    elif key == KeyCode(char="s"):
        IS_HUMAN_TALKING = True
        command = record_input()
        print(command)
        if command:
            HUMAN_REWARD_SIGNAL = language_model.get_score(command)
        else:
            print("Spike could not understand!")
            HUMAN_REWARD_SIGNAL = 0
        print(HUMAN_REWARD_SIGNAL)
        IS_HUMAN_TALKING = False


def record_input():
    try:
        with microphone as source:
            speech_recognizer.adjust_for_ambient_noise(source)
            audio = speech_recognizer.listen(source, timeout=1.5, phrase_time_limit=4)  # read the entire audio file
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        return speech_recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return False
    except sr.RequestError as e:
        return False
    except sr.WaitTimeoutError as e:
        return False


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


def target_update(local_network, target_network):
    with torch.no_grad():
        # Soft update the weights of the actor target network
        for target_param, param in zip(target_network.parameters(), local_network.parameters()):
            target_param.copy_(
                target_param * (1.0 - SOFT_UPDATE_WEIGHT) + param * SOFT_UPDATE_WEIGHT
            )

def hard_update(local_network, target_network):
    for target_param, param in zip(target_network.parameters(), local_network.parameters()):
        target_param.data.copy_(param.data)


if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    nb_actions = environment.action_space.n
    nb_states = environment.observation_space.shape[0]
    hidden_state = 32
    loss = torch.nn.MSELoss()

    speech_recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=1)
    language_model = BertTransformerVerbalReward('LM/bert_textclass.pt').cuda()
    keyboard_listener = keyboard.Listener(on_press=reward_input_handler,)
    keyboard_listener.start()

    print("Training Agent")
    total_pd = pd.DataFrame()
    total_verification_mean = []
    for i in range(4):
        STEPS_DONE = 0
        reward_estimator = RewardNetwork(nb_states, hidden_state, nb_actions).to(device)
        target_reward_estimator = RewardNetwork(
            nb_states, hidden_state, nb_actions
        ).to(device)
        hard_update(reward_estimator, target_reward_estimator)

        optim = torch.optim.AdamW(lr=0.005, params=reward_estimator.parameters())
        print(f"----------------Trial {i} starting--------------")
        time.sleep(5)
        (
            reward_estimator,
            feedback_counter_positive,
            feedback_counter_negative,
            reward_counter,
        ) = train(
            environment,
            reward_estimator,
            target_reward_estimator,
            loss,
            optim,
            15,
            100,
            nb_actions,
            use_hyperball=True,
            art_states=10,
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

    total_pd.to_csv("TAMER_H20_LM_target_net_experiment_extra.csv")
    pd.DataFrame(total_verification_mean).to_csv(
        "CS7648 Project/data_might_be_useless/TAMER_HS20_LM_target_net_verification_extra.csv")
