from robot_manager import RobotManager
from robomaster import robot
from tamer_model import RewardNetwork
import numpy as np
import torch
import time
import argparse
from pynput import keyboard, mouse
from pynput.keyboard import KeyCode
from pynput.mouse import Button
import random
from collections import Counter
from utils import calculate_reward
import pandas as pd
import math
import speech_recognition as sr

from language_model.model import BertTransformerVerbalReward

HUMAN_REWARD_SIGNAL = 0.0
IS_HUMAN_TALKING = False
TERMINATE = False
SOFT_UPDATE_WEIGHT = .01
# X_MAX = 210
# X_MIN = 70
# Y_MAX = 150
# Y_MIN = -28
SOFT_UPDATE_WEIGHT = .01
TRUST_DECAY_START = 0.9
TRUST_DECAY_END = 0.05
TRUST_DECAY_RATE = 100
STEPS_DONE = 0

def train(
    manager: RobotManager,
    reward_network: RewardNetwork,
    target_reward_network: RewardNetwork,
    loss_criterion,
    optimizer,
    starting_epoch: int,
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
    reward_buffer = []
    window = []
    feedback_counter_positive = Counter()
    feedback_counter_negative = Counter()
    reward_counter = Counter()

    for epoch in range(starting_epoch, starting_epoch + epochs + 1):
        if epoch % 5 == 0:
            print("Saving checkpoint...")
            torch.save(
                {
                    "reward_network": reward_network.state_dict(),
                    "reward_optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                f"TAMER_chkp/chkp-{epoch}.pt",
            )

        TERMINATE = False

        print("Resetting Robot Arm")
        manager.reset_arm()
        print(f"Starting Epoch: {epoch}.")

        step_counter = 0
        HUMAN_REWARD_SIGNAL = 0.0
        curr_accumulated_reward = 0
        feedback_counter_positive[epoch] = 0
        feedback_counter_negative[epoch] = 0
        reward_counter[epoch] = 0

        while step_counter < max_length:
            if IS_HUMAN_TALKING:
                time.sleep(8)
            STEPS_DONE += 1
            step_counter += 1
            state = manager.get_state()
            reward_predictions = reward_network(state)
            best_action = int(torch.argmax(reward_predictions).item())

            if random.random() > 0.95:
                best_action = random.randint(0, num_actions - 1)

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
                    [reward_buffer[-1]], loss_criterion, optimizer, reward_network, use_hyperball, art_states, target_network=target_reward_network
                )
                window = []
                if HUMAN_REWARD_SIGNAL > 0.0:
                    feedback_counter_positive[epoch] += 1
                else:
                    feedback_counter_negative[epoch] += 1
                HUMAN_REWARD_SIGNAL = 0.0
            else:
                # sample from buffer
                if len(reward_buffer) > minibatch_size:
                    window_sample = random.choices(reward_buffer, k=minibatch_size)
                else:
                    window_sample = reward_buffer

                if window_sample:
                    update_weights(
                        window_sample, loss_criterion, optimizer, reward_network, use_hyperball, art_states, target_network=target_reward_network
                    )

            curr_accumulated_reward += calculate_reward(
                TERMINATE, state, np.zeros((1, 4))
            ).item()

            if TERMINATE:
                print("This epoch has been aborted.")
                break

            take_action(best_action, manager)
            time.sleep(0.5)

        reward_counter[epoch] = curr_accumulated_reward

    return (
        reward_network,
        feedback_counter_positive,
        feedback_counter_negative,
        reward_counter,
    )

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

def verify(trained_agent: RewardNetwork, manager: RobotManager):
    global TERMINATE
    global STEPS_DONE
    trained_agent.eval()
    reward_total = 0
    for _ in range(10):
        reward_epoch = 0
        manager.reset_arm()
        state = manager.get_state()
        for i in range(50):
            action = int(trained_agent(state).argmax(dim=1).item())
            take_action(action, manager)
            state = manager.get_state()
            reward_epoch += calculate_reward(
                TERMINATE, state, np.zeros((1, 4))
            ).item()
            if TERMINATE or i >= 49:
                print(f"Trial reward:{reward_epoch}")
                reward_total += reward_epoch
                break
        time.sleep(0.2)
    print(f"Average Reward: {reward_total/10}")
    return reward_total / 10


def update_weights(
    window_sample, loss_criterion, optimizer, reward_network, use_hyperball, art_states, target_network=None
):
    optimizer.zero_grad()
    total_loss = torch.zeros((1,))
    for sample, human_reward, credit in window_sample:
        for state, action in sample:
            reward_predictions = reward_network(state)
            target = reward_predictions.clone()
            curr_reward = target[0][action]
            mask = target == curr_reward
            reward_signal = torch.ones_like(target) * human_reward
            target = torch.where(mask, reward_signal, torch.zeros_like(reward_signal))
            total_loss += loss_criterion(reward_predictions, target) * credit
            if use_hyperball:
                for _ in range(art_states):
                    art_state = generate_artificial_state(state)
                    reward_predictions = reward_network(art_state)
                    target = reward_predictions.clone()
                    curr_reward = target[0][action]
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


def generate_artificial_state(state, scale=0.1):
    new_state = []
    for i, element in enumerate(state):
        if i < 3:
            new_state.append(np.random.normal(element, np.abs(element) * scale))
        else:
            new_state.append(element)

    return torch.from_numpy(new_state[0].astype(np.float32)).unsqueeze(dim=0)


def take_action(action: int, manager: RobotManager):
    action_vector = np.zeros((1, 4))
    if action == 0:  # forward ee movement
        print("Forward EE")
        action_vector[0][0] = 22.0
        manager.execute_action(action_vector)
    elif action == 1:  # backward ee movement
        print("Backward EE")
        action_vector[0][0] = -22.0
        manager.execute_action(action_vector)
    elif action == 2:  # upward ee movement
        print("Upward EE")
        action_vector[0][1] = 22.0
        manager.execute_action(action_vector)
    elif action == 3:  # downward ee movement
        print("Downward EE")
        action_vector[0][1] = -22.0
        manager.execute_action(action_vector)
    elif action == 4:  # ccw base rotation
        print("CCW Base")
        action_vector[0][2] = 22.0
        manager.execute_action(action_vector)
    elif action == 5:  # cw base rotation
        print("CW Base")
        action_vector[0][2] = -22.0
        manager.execute_action(action_vector)
    elif action == 6:  # open gripper
        print("Open Gripper")
        action_vector[0][3] = 1.0
        manager.execute_action(action_vector)
    elif action == 7:  # close gripper
        print("Close Gripper")
        action_vector[0][3] = -1.0
        manager.execute_action(action_vector)


def reward_input_handler(key):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE
    global IS_HUMAN_TALKING
    if key == KeyCode(char="s"):
        IS_HUMAN_TALKING = True
        command = record_input()
        print(command)
        if command:
            HUMAN_REWARD_SIGNAL = record_input()
        else:
            print("Spike could not understand!")
            HUMAN_REWARD_SIGNAL = 0
        print(HUMAN_REWARD_SIGNAL)
        IS_HUMAN_TALKING = False
    elif key == KeyCode(char="q"):
        HUMAN_REWARD_SIGNAL = -10
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-data-path",
        help="where to save the ground truth reward data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-model-path", help="where to save the model", type=str, required=True
    )
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--language-model", type=str, default=None)
    parser.add_argument(
        "--num-inputs",
        help="number of input parameters to the networks",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-outputs",
        help="number of actions for the actor network",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--hidden-size",
        help="number of nodes in each hidden layer",
        type=int,
        default=256,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trajectory-length", type=int, default=100)
    parser.add_argument(
        "--cuda", help="Set to True in order to use GPU", type=bool, default=False
    )

    args = parser.parse_args()

    language_model = BertTransformerVerbalReward(args.language_model).cuda() if args.language_model else None
    reward_estimator = RewardNetwork(
        args.num_inputs, args.hidden_size, args.num_outputs
    )
    target_reward_estimator = RewardNetwork(
        args.num_inputs, args.hidden_size, args.num_outputs
    )

    hard_update(reward_estimator, target_reward_estimator)

    optim = torch.optim.AdamW(
        lr=args.learning_rate, params=reward_estimator.parameters()
    )
    epoch = 0
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        reward_estimator.load_state_dict(checkpoint["reward_network"])
        optim.load_state_dict(checkpoint["reward_optimizer"])
        epoch = checkpoint["epoch"]
        print("Loaded from checkpoint: {}".format(args.checkpoint_path))
    loss = torch.nn.MSELoss()

    keyboard_listener = keyboard.Listener(on_press=reward_input_handler,)
    keyboard_listener.start()

    print("Training Agent")
    total_pd = pd.DataFrame()
    total_verification_mean = []

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap", proto_type="udp")
    manager = RobotManager(ep_robot)
    speech_recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=1)
    time.sleep(2)

    (
        learned_reward,
        feedback_counter_positive,
        feedback_counter_negative,
        reward_counter,
    ) = train(
        manager,
        reward_estimator,
        target_reward_estimator,
        loss,
        optim,
        epoch,
        args.epochs,
        args.trajectory_length,
        args.num_outputs,
        use_hyperball=True,
        art_states=100
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
    final_pd["trial"] = 0
    total_pd = pd.concat([total_pd, final_pd])
    print("Running Verification")
    total_verification_mean.append(verify(reward_estimator, manager))
    time.sleep(1.0)

    total_pd.to_csv("spike_TAMER_balls_and_whistles_train.csv")
    pd.DataFrame(total_verification_mean).to_csv("spike_TAMER_balls_and_whistles_val.csv")
    ep_robot.close()
    keyboard_listener.stop()
