from robot_manager import RobotManager
from robomaster import robot
from tamer_model import RewardNetwork
from scipy.stats import gamma
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

from language_model.model import BertTransformerVerbalReward

HUMAN_REWARD_SIGNAL = 0.0
IS_HUMAN_TALKING = False
TERMINATE = False
X_MAX = 210
X_MIN = 70
Y_MAX = 150
Y_MIN = -28


def train(
    manager: RobotManager,
    reward_network: RewardNetwork,
    loss_criterion,
    optimizer,
    starting_epoch: int,
    epochs: int,
    max_length: int,
    num_actions: int,
    window_size: int = 10,
    minibatch_size: int = 16,
):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE
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
                time.sleep(7)

            step_counter += 1
            state = manager.get_state()
            reward_predictions = reward_network(state)
            best_action = int(torch.argmax(reward_predictions).item())

            if random.random() > 0.95:
                best_action = random.randint(0, num_actions)

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
                    [reward_buffer[-1]], loss_criterion, optimizer, reward_network
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
                        window_sample, loss_criterion, optimizer, reward_network
                    )

            curr_accumulated_reward += calculate_reward(
                TERMINATE, state, np.zeros((1, 4))
            )

            if TERMINATE:
                print("This epoch has been aborted.")
                break

            take_action(best_action, manager)

        reward_counter[epoch] = curr_accumulated_reward

    return (
        reward_network,
        feedback_counter_positive,
        feedback_counter_negative,
        reward_counter,
    )


def verify(trained_agent: RewardNetwork, manager: RobotManager):
    global TERMINATE
    trained_agent.eval()
    reward_total = 0
    for _ in range(10):
        reward_epoch = 0
        manager.reset_arm()
        state = manager.get_state()
        for _ in range(100):
            action = trained_agent(state).argmax(dim=0)
            take_action(action, manager)
            reward_epoch += calculate_reward(TERMINATE, state, np.zeros((1, 4)))
            if TERMINATE:
                print(f"Trial reward:{reward_epoch}")
                reward_total += reward_epoch
                break
        time.sleep(0.2)
    print(f"Average Reward: {reward_total/10}")


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
    for i, element in enumerate(state):
        if i < 3:
            new_state.append(np.random.normal(element, np.abs(element) * scale))
        else:
            new_state.append(element)
    return torch.tensor(new_state)


def take_action(action: int, manager: RobotManager):
    action_vector = np.zeros((1, 4))
    end_effector_x, end_effector_y = manager.ee_body_pose[0][:-1]
    if action == 0:  # forward ee movement
        if end_effector_x + 10.0 < X_MAX:
            action_vector[0][0] = 10.0
            manager.execute_action(action_vector)
        print("Forward EE")
    elif action == 1:  # backward ee movement
        if end_effector_x - 10.0 > X_MIN:
            action_vector[0][0] = -10.0
            manager.execute_action(action_vector)
        print("Backward EE")
    elif action == 2:  # upward ee movement
        if end_effector_y + 10 < Y_MAX:
            action_vector[0][1] = 10.0
            manager.execute_action(action_vector)
        print("Upward EE")
    elif action == 3:  # downward ee movement
        if end_effector_y - 10 > Y_MIN:
            action_vector[0][1] = -10.0
            manager.execute_action(action_vector)
        print("Downward EE")
    elif action == 4:  # ccw base rotation
        action_vector[0][2] = 15.0
        manager.execute_action(action_vector)
        print("CCW Base")
    elif action == 5:  # cw base rotation
        action_vector[0][2] = -15.0
        manager.execute_action(action_vector)
        print("CW Base")
    elif action == 4:  # open gripper
        action_vector[0][3] = 1.0
        manager.execute_action(action_vector)
        print("Open Gripper")
    elif action == 5:  # close gripper
        action_vector[0][3] = -1.0
        manager.execute_action(action_vector)
        print("Close Gripper")


def reward_input_handler(key):
    global HUMAN_REWARD_SIGNAL
    global TERMINATE
    global IS_HUMAN_TALKING
    if key == KeyCode(char="s"):
        IS_HUMAN_TALKING = True
        command = manager.transcribe_audio()
        print(command)
        if command:
            HUMAN_REWARD_SIGNAL = language_model.get_score(command)
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


def mouse_reward_handler(x, y, button, pressed):
    global HUMAN_REWARD_SIGNAL
    if button == Button.left and pressed:
        HUMAN_REWARD_SIGNAL = 1
    elif button == Button.right and pressed:
        HUMAN_REWARD_SIGNAL = -1


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

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap", proto_type="udp")
    manager = RobotManager(ep_robot)
    language_model = None
    reward_estimator = RewardNetwork(
        args.num_inputs, args.hidden_size, args.num_outputs
    )
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

    mouse_listener = mouse.Listener(on_click=mouse_reward_handler)
    mouse_listener.start()

    print("Training Agent")
    total_pd = pd.DataFrame()
    total_verification_mean = []

    (
        learned_reward,
        feedback_counter_positive,
        feedback_counter_negative,
        reward_counter,
    ) = train(
        manager,
        reward_estimator,
        loss,
        optim,
        epoch,
        args.epochs,
        args.trajectory_length,
        args.num_outputs,
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

    total_pd.to_csv("no_hyperball_experiment.csv")
    pd.DataFrame(total_verification_mean).to_csv("no_hyperball_verification.csv")
    ep_robot.close()
    keyboard_listener.stop()