from robot_manager import RobotManager
from robomaster import robot
from tamer_model import RewardNetwork
from scipy.stats import gamma
import numpy as np
import torch
import time
import argparse


def train(manager: RobotManager, reward_network: RewardNetwork, loss_criterion, optimizer, epochs: int,
          max_length: int):
    for epoch in range(epochs):
        step_counter = 0
        human_reward = 0.0
        creditor = {}
        for step in range(max_length):
            step_counter += 1
            state = manager.get_state()
            reward_predictions = reward_network(state)
            best_action = int(torch.argmax(reward_predictions).item())
            creditor[step_counter] = (reward_predictions, best_action, gamma.pdf(step_counter, 2.0, 0.5, 0.28))

            if step_counter % 5 == 0:
                print("Action performed: {}".format(best_action))
                human_reward = float(input("Please enter reward signal (-5 - 5): "))

            if human_reward != 0.0:
                update_weights(human_reward, creditor, loss_criterion, optimizer)
                human_reward = 0.0

            take_action(best_action, manager)

    return reward_network


def update_weights(reward_signal: float, creditor, loss_criterion, optimizer):
    for predicted_reward, best_action, credit in creditor.values():
        target = predicted_reward.clone()
        target[0, best_action] = reward_signal
        step_loss = loss_criterion(credit * predicted_reward, target)
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

    learned_reward = train(manager, reward_estimator, loss, optim, args.epochs, args.trajectory_length)
