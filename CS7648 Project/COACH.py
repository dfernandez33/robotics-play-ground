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
from utils import calculate_reward
import pandas as pd
from rl_models import PolicyNetwork

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
    accumulated_rewards = []

    window = []
    eligibility_replay_buffer = []

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
        while step_counter < max_length:
            if IS_HUMAN_TALKING:
                time.sleep(7)

            if step_counter:
                window.append((state, best_action, p_best_action, HUMAN_REWARD_SIGNAL))

            if HUMAN_REWARD_SIGNAL != 0.0:
                eligibility_replay_buffer.append(window)
                window = []
                HUMAN_REWARD_SIGNAL = 0.0

            step_counter += 1
            state = manager.get_state()
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

            curr_accumulated_reward += calculate_reward(
                TERMINATE, state, np.zeros((1, 4))
            )

            if TERMINATE:
                print("This epoch has been aborted.")
                break

            take_action(best_action, manager)

        print(f"Accumulated_reward over epoch {epoch}: {curr_accumulated_reward}")
        accumulated_rewards.append(curr_accumulated_reward)

    return reward_network, accumulated_rewards


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
    elif action == 6:  # open gripper
        action_vector[0][3] = 1.0
        manager.execute_action(action_vector)
        print("Open Gripper")
    elif action == 7:  # close gripper
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
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trajectory-length", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=16)
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
    policy_network = PolicyNetwork(args.num_inputs, args.hidden_size, args.num_outputs)
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

    learned_reward, accumulated_rewards = train(
        manager,
        reward_estimator,
        policy_network,
        loss,
        optim,
        epoch,
        args.epochs,
        args.trajectory_length,
        args.minibatch_size,
    )

    torch.save(
        {"reward_network": learned_reward.state_dict(),}, f"{args.save_model_path}"
    )
    print(accumulated_rewards)
    pd.DataFrame(accumulated_rewards).to_csv(args.save_data_path)

    ep_robot.close()
    keyboard_listener.stop()
