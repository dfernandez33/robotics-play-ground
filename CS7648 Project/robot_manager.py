from robomaster import robot
from enum import Enum
import time
import random
import numpy as np
import ctypes
import math
import torch


class RobotManager:
    def __init__(self, ep_bot: robot):
        self.ep_robot = ep_bot
        self.battery_level = 100
        self.gripper_state = GripperState.CLOSED
        self.ee_offset = 76.2
        self.ee_body_pose = np.ones((1, 3))
        self.chassis_attitude = np.ones((1, 3))
        self.chassis_position = np.ones((1, 3))
        self.ep_robot.robotic_arm.sub_position(freq=20, callback=self.__arm_position_handler)
        self.ep_robot.chassis.sub_attitude(freq=20, callback=self.__chassis_attitude_handler)
        self.ep_robot.chassis.sub_position(freq=20, callback=self.__chassis_position_handler)
        self.ep_robot.battery.sub_battery_info(freq=1, callback=self.__battery_handler)

    def reset_arm(self):
        # Choose random starting configuration
        x_pose = random.randrange(78, 122)
        y_pose = random.randrange(57, 110)
        rotation = random.randrange(0, 360)

        self.ep_robot.gripper.open()
        time.sleep(2)
        self.ep_robot.gripper.pause()
        self.ep_robot.robotic_arm.moveto(x_pose, y_pose).wait_for_completed()
        self.ep_robot.gripper.close()
        time.sleep(2)
        self.ep_robot.gripper.pause()
        self.gripper_state = GripperState.CLOSED
        self.ep_robot.chassis.move(z=self.chassis_attitude[0][0]).wait_for_completed()  # rotation to starting position
        self.ep_robot.chassis.move(z=rotation).wait_for_completed()  # rotate to random position

    def execute_action(self, action: np.ndarray):
        arm_x_movement = round(action[0][0], 2)
        arm_y_movement = round(action[0][1], 2)
        chassis_rotation = round(action[0][2], 2)
        gripper = round(action[0][3], 2)

        self.ep_robot.robotic_arm.move(arm_x_movement, arm_y_movement).wait_for_completed()
        self.ep_robot.chassis.move(z=chassis_rotation).wait_for_completed()
        # correct drift due to rotation
        # chassis_rotation_rad = math.radians(chassis_rotation)
        # chassis_x = self.chassis_position[0][0]
        # chassis_y = self.chassis_position[0][1]
        # x_offset = chassis_x*math.cos(chassis_rotation_rad) + chassis_y*math.sin(chassis_rotation_rad)
        # y_offset = -math.sin(chassis_rotation_rad)*chassis_x + math.cos(chassis_rotation_rad)*chassis_y
        # self.ep_robot.chassis.move(x=-x_offset, y=-y_offset).wait_for_completed()
        if gripper > 0 and self.gripper_state != GripperState.OPEN:
            self.ep_robot.gripper.open()
            time.sleep(2)
            self.ep_robot.gripper.pause()
            self.gripper_state = GripperState.OPEN
            return False
        elif gripper < 0 and self.gripper_state != GripperState.CLOSED:
            self.ep_robot.gripper.close()
            time.sleep(2)
            self.ep_robot.gripper.pause()
            self.gripper_state = GripperState.CLOSED
            return True
        return False

    def get_state(self):
        rotation = math.radians(self.chassis_attitude[0][0])
        bTs = np.array([[math.cos(rotation), 0, math.sin(rotation), -304.8],
                       [0, 1, 0, -76.2],
                       [-math.sin(rotation), 0, math.cos(rotation), 0],
                       [0, 0, 0, 1]])
        ee_body = np.array([[self.ee_body_pose[0][0]], [self.ee_body_pose[0][1]], [self.ee_body_pose[0][2]], [1]])
        ee_spatial = bTs @ ee_body
        state = torch.tensor([[ee_spatial[0][0], ee_spatial[1][0], ee_spatial[2][0],
                              1.0 if self.gripper_state == GripperState.OPEN else 0.0]])
        return state.float()

    def __battery_handler(self, battery_info):
        self.battery_level = battery_info

    def __arm_position_handler(self, sub_info):
        pos_x, pos_y = sub_info
        pos_y = ctypes.c_int32(pos_y).value
        self.ee_body_pose = np.array([[pos_x + self.ee_offset, pos_y, 0]])

    def __chassis_attitude_handler(self, sub_info):
        yaw, pitch, roll = sub_info
        self.chassis_attitude = np.array([[yaw, pitch, roll]])

    def __chassis_position_handler(self, pos_info):
        x, y, z = pos_info
        self.chassis_position = np.array([[x, y, z]])


class GripperState(Enum):
    OPEN = 1
    CLOSED = 2
