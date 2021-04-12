from robot_manager import RobotManager
from robomaster import robot
from tamer_model import RewardNetwork
from scipy.stats import gamma
import numpy as np
import torch
import time
import argparse
from pynput import keyboard
from pynput.keyboard import KeyCode

HUMAN_REWARD_SIGNAL = 0.0


def train(manager: RobotManager, reward_network: RewardNetwork, loss_criterion, optimizer, epochs: int,
          max_length: int):
    global HUMAN_REWARD_SIGNAL
    for epoch in range(epochs):
        print("Starting new training epoch")
        manager.reset_arm()
        step_counter = 0
        HUMAN_REWARD_SIGNAL = 0.0
        creditor = {}
        for step in range(max_length):
            step_counter += 1
            state = manager.get_state()
            reward_predictions = reward_network(state)
            best_action = int(torch.argmax(reward_predictions).item())
            creditor[step_counter] = (reward_predictions, best_action, time.time())

            if HUMAN_REWARD_SIGNAL != 0.0:
                update_weights(HUMAN_REWARD_SIGNAL, time.time(), creditor, loss_criterion, optimizer)
                HUMAN_REWARD_SIGNAL = 0.0
                creditor = {}

            take_action(best_action, manager)

    return reward_network


def update_weights(reward_signal: float, human_time: float, creditor, loss_criterion, optimizer):
    for predicted_reward, best_action, action_time in creditor.values():
        credit = gamma.pdf((human_time - action_time), 2.0, 0.0, 0.28)
        target = torch.zeros_like(predicted_reward)
        target[0, best_action] = reward_signal
        credited_predictions = credit * predicted_reward
        step_loss = loss_criterion(credited_predictions, target)
        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()


def take_action(action: int, manager: RobotManager):
    action_vector = np.zeros((1, 4))
    if action == 0:  # forward ee movement
        action_vector[0][0] = 10.0
        manager.execute_action(action_vector)
    elif action == 1:  # backward ee movement
        action_vector[0][0] = -10.0
        manager.execute_action(action_vector)
    elif action == 2:  # upward ee movement
        action_vector[0][1] = 10.0
        manager.execute_action(action_vector)
    elif action == 3:  # downward ee movement
        action_vector[0][1] = -10.0
        manager.execute_action(action_vector)
    elif action == 4:  # ccw base rotation
        action_vector[0][2] = 15.0
        manager.execute_action(action_vector)
    elif action == 5:  # cw base rotation
        action_vector[0][2] = -15.0
        manager.execute_action(action_vector)
    elif action == 6:  # open gripper
        action_vector[0][3] = 1.0
        manager.execute_action(action_vector)
    elif action == 7:  # close gripper
        action_vector[0][3] = -1.0
        manager.execute_action(action_vector)


def reward_input_handler(key):
    global HUMAN_REWARD_SIGNAL
    if key == KeyCode(char='1'):
        HUMAN_REWARD_SIGNAL = -5.0
    elif key == KeyCode(char='2'):
        HUMAN_REWARD_SIGNAL = -4.0
    elif key == KeyCode(char='3'):
        HUMAN_REWARD_SIGNAL = -3.0
    elif key == KeyCode(char='4'):
        HUMAN_REWARD_SIGNAL = -2.0
    elif key == KeyCode(char='5'):
        HUMAN_REWARD_SIGNAL = -1.0
    elif key == KeyCode(char='6'):
        HUMAN_REWARD_SIGNAL = 0.0
    elif key == KeyCode(char='7'):
        HUMAN_REWARD_SIGNAL = 1.0
    elif key == KeyCode(char='8'):
        HUMAN_REWARD_SIGNAL = 2.0
    elif key == KeyCode(char='9'):
        HUMAN_REWARD_SIGNAL = 3.0
    elif key == KeyCode(char='0'):
        HUMAN_REWARD_SIGNAL = 4.0
    elif key == KeyCode(char='-'):
        HUMAN_REWARD_SIGNAL = 5.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', help="where to save the model", type=str, required=True)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--num-inputs', help="number of input parameters to the networks", type=int, default=10)
    parser.add_argument('--num-outputs', help="number of actions for the actor network", type=int, default=10)
    parser.add_argument('--hidden-size', help="number of nodes in each hidden layer", type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trajectory-length', type=int, default=100)
    parser.add_argument('--cuda', help="Set to True in order to use GPU", type=bool, default=False)

    args = parser.parse_args()

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type='ap', proto_type='udp')
    manager = RobotManager(ep_robot)

    reward_estimator = RewardNetwork(args.num_inputs, args.hidden_size, args.num_outputs)
    optim = torch.optim.AdamW(lr=args.learning_rate, params=reward_estimator.parameters())
    loss = torch.nn.MSELoss()

    keyboard_listener = keyboard.Listener(
        on_press=reward_input_handler,
    )
    keyboard_listener.start()

    learned_reward = train(manager, reward_estimator, loss, optim, args.epochs, args.trajectory_length)

    ep_robot.close()
    keyboard_listener.stop()
