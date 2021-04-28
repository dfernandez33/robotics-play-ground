from robomaster import robot
from enum import Enum
import time
import random
import numpy as np
import ctypes
import math
import torch
import speech_recognition as sr
from os import path


class RobotManager:
    def __init__(self, ep_bot: robot):
        # robot control properties
        self.ep_robot = ep_bot
        self.battery_level = 100
        self.gripper_state = GripperState.CLOSED
        self.ee_offset = 76.2
        self.arm_y_lower_limit = -20
        self.ee_body_pose = np.ones((1, 3))
        self.chassis_attitude = np.ones((1, 3))
        self.chassis_position = np.ones((1, 3))
        self.ep_robot.robotic_arm.sub_position(freq=20, callback=self.__arm_position_handler)
        self.ep_robot.chassis.sub_attitude(freq=20, callback=self.__chassis_attitude_handler)
        self.ep_robot.chassis.sub_position(freq=20, callback=self.__chassis_position_handler)
        self.ep_robot.battery.sub_battery_info(freq=1, callback=self.__battery_handler)

        # speech properties
        self.speech_recognizer = sr.Recognizer()

    def reset_arm(self):
        self.ep_robot.play_sound(robot.SOUND_ID_RECOGNIZED).wait_for_completed()
        # Choose random starting configuration
        x_pose = random.randrange(78, 122)
        y_pose = random.randrange(57, 110)
        rotation = random.randrange(-180, 180)

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
        self.ep_robot.play_sound(robot.SOUND_ID_RECOGNIZED).wait_for_completed()

    def execute_action(self, action: np.ndarray):
        arm_x_movement = round(action[0][0], 2)
        arm_y_movement = round(action[0][1], 2)
        chassis_rotation = round(action[0][2], 2)
        gripper = round(action[0][3], 2)

        valid_y_movement = self.__check_arm_y_movement(arm_y_movement)
        self.ep_robot.robotic_arm.move(arm_x_movement, valid_y_movement).wait_for_completed()
        self.ep_robot.chassis.move(z=chassis_rotation, z_speed=45).wait_for_completed()

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
        rotation = math.radians(-self.chassis_attitude[0][0])
        bTs = np.array([[math.cos(rotation), 0, math.sin(rotation), -304.8],
                       [0, 1, 0, -114.3],
                       [-math.sin(rotation), 0, math.cos(rotation), 0],
                       [0, 0, 0, 1]])
        ee_body = np.array([[self.ee_body_pose[0][0]], [self.ee_body_pose[0][1]], [self.ee_body_pose[0][2]], [1]])
        ee_spatial = bTs @ ee_body
        state = torch.tensor([[ee_spatial[0][0], ee_spatial[1][0], ee_spatial[2][0],
                              1.0 if self.gripper_state == GripperState.OPEN else 0.0]])
        return state.float()

    def set_state(self, state):
        x_pose, y_pose, rotation, gripper_status = state
        self.ep_robot.chassis.move(z=rotation).wait_for_completed() 
        self.ep_robot.robotic_arm.moveto(x_pose, y_pose).wait_for_completed()
        if gripper_status == GripperState.CLOSED:
            self.ep_robot.gripper.close()
            time.sleep(2)
            self.gripper_state = GripperState.CLOSED
        elif gripper_status == GripperState.OPEN:
            self.ep_robot.gripper.open()
            time.sleep(2)
            self.gripper_state = GripperState.OPEN
        else:
            print('ERROR! Gripper state unk.')

        self.ep_robot.play_sound(robot.SOUND_ID_RECOGNIZED).wait_for_completed()

    def transcribe_audio(self, duration=3):
        print("Listening for verbal reward signal")
        self.ep_robot.camera.record_audio(save_file="verbal_reward.wav", seconds=duration, sample_rate=16000)
        audio_file = path.join(path.dirname(path.realpath(__file__)), "verbal_reward.wav")
        with sr.AudioFile(audio_file) as source:
            audio = self.speech_recognizer.record(source)  # read the entire audio file

        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            return self.speech_recognizer.recognize_google(audio)
        # TODO: handle exception properly, in case of failure prompt user to provide signal again.
        except sr.UnknownValueError:
            return False
        except sr.RequestError as e:
            return False

    def __check_arm_y_movement(self, y_movement):
        valid_y_movement = y_movement
        if y_movement < 0 and self.ee_body_pose[0][1] + y_movement <= self.arm_y_lower_limit:
            valid_y_movement = -abs(self.ee_body_pose[0][1] - self.arm_y_lower_limit)
        return valid_y_movement

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
