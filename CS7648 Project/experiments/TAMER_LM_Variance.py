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

from language_model.model import BertTransformerVerbalReward

HUMAN_REWARD_SIGNAL = 0.0
IS_HUMAN_TALKING = False
TERMINATE = False
X_MAX = 210
X_MIN = 70
Y_MAX = 150
Y_MIN = -28

STATE_ACTION_DICT = {}


def verify(states: list, manager: RobotManager):
    global TERMINATE
    global HUMAN_REWARD_SIGNAL

    print("------------Starting Training Loop--------------")

    # sample states with replacement
    state_dict = {}
    action_loop = [1, 3, 4, 2, 1, 1, 5, 5, 6, 2, 7]
    
    for _ in range(20):
        curr_state = np.random.choices(states, size=1)

        print("-------------Resetting Robot-------------")
        manager.set_state(curr_state)

        if curr_state not in state_dict:
            state_dict[curr_state] = []
        
        for action in action_loop:
            take_action(action, manager)
            time.sleep(1)
            state_dict[curr_state].append(HUMAN_REWARD_SIGNAL)
    
    return state_dict


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


def generate_random_states():
    return [
        (85, 98, 56, 0),
        (94, 105, -133, 0),
        (95, 86, 177, 0),
        (79, 100, -73, 0),
        (81, 72, 155, 0),
    ]
    


if __name__ == "__main__":
    language_model = language_model.BertTransformerVerbalReward('LM/bert_stccalss.pt')
    state_num = 5

    keyboard_listener = keyboard.Listener(on_press=reward_input_handler,)
    keyboard_listener.start()

    print("Training Agent")

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap", proto_type="udp")
    manager = RobotManager(ep_robot)
    
    random_states = generate_random_states()
    results = verify(random_states, manager)

    #TODO: make dataframe and CSV dump for results

    
    robot.close()
    keyboard_listener.stop()
