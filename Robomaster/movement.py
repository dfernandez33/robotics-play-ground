import socket
import time
from threading import Lock, Thread
from Robomaster.basecommandmodule import BaseCommandModule
from typing import Dict
from timeit import default_timer


class Movement(BaseCommandModule):

    def __init__(self, control_socket: socket.socket, socket_lock: Lock):
        super(Movement, self).__init__(control_socket, socket_lock)
        self.pose_capture_thread = Thread(target=self.__get_pose, daemon=True)
        self.pose_mutex = Lock()
        self.position = {'x': 0, 'y': 0, 'z': 0}
        self.speed = {'x': 0, 'y': 0, 'z': 0, 'w1': 0, 'w2': 0, 'w3': 0, 'w4': 0}
        self.attitude = {'pitch': 0, 'roll': 0, 'yaw': 0}
        self.pose_capture_thread.start()

    def drive(self, x_speed=0.0, y_speed=0.0, rotation_speed=0.0, duration=2.0, await_response=True) -> bool:
        """
        Commands the robomaster to drive with the assigned velocities for duration seconds
        :param x_speed: forward/backward velocity in m/s
        :param y_speed: left/right velocity in m/s
        :param rotation_speed: rotation speed in degrees/s
        :param duration: how long to maintain velocities for
        :param await_response: whether to wait for the robot to respond with execution status
        :return: whether the command executed correctly or not
        """
        cmd = 'chassis speed x {} y {} z {}'.format(x_speed, y_speed, rotation_speed)
        response = super().send_command(cmd, await_response)
        if response != 'ok':
            return False
        time.sleep(duration)
        return self.stop(await_response)

    def stop(self, await_response=True) -> bool:
        """
        Stops the robots motion in all directions
        :return: whether the command executed correctly or not
        """
        cmd = 'chassis wheel w1 0 w2 0 w3 0 w4 0'
        response = super().send_command(cmd, await_response)
        return response == 'ok'

    def drive_wheels(self, w1=0.0, w2=0.0, w3=0.0, w4=0.0, duration=2.0, await_response=True) -> bool:
        """
        Control the speed of the four wheel individually, in rpm.
        :param w1: Right front wheel speed
        :param w2: Left front wheel speed
        :param w3: Right rear wheel speed
        :param w4: Left rear wheel speed
        :param duration: how long to execute command for
        :return: whether the command executed correctly or not
        """
        cmd = 'chassis wheel w1 {} w2 {} w3 {} w4 {}'.format(w1, w2, w3, w4)
        response = super().send_command(cmd, await_response)
        if response != 'ok':
            return False
        time.sleep(duration)
        return self.stop(await_response)

    def move_to_pose(self, distance_x=0.0, distance_y=0.0, rotation=0.0, speed_xy=.2, rotation_speed=30,
                     await_arrival=True, time_out=5.0) -> bool:
        """
        Move to a spcified pose relative to the robot's coordinate frame
        :param await_response: If the program should wait for the response before returning
        :param distance_x: x-axial distance in m
        :param distance_y: y-axial distance in m
        :param rotation: z-axial distance in degrees
        :param speed_xy: speed to move at in m/s
        :param rotation_speed: speed to rotate at in degrees/s
        :return: whether the command executed correctly or not
        """
        initial_position = self.get_position()
        goal_position = {'x': initial_position['x'] + distance_x,
                         'y': initial_position['y'] + distance_y,
                         'z': initial_position['z'] + rotation}
        cmd = 'chassis move x {} y {} z {} vxy {} vz {}'.format(distance_x, distance_y, rotation, speed_xy, rotation_speed)
        response = super().send_command(cmd)
        if await_arrival:
            movement_start = default_timer()
            curr_position = self.get_position()
            while abs(goal_position['x'] - curr_position['x']) > .1 or abs(goal_position['y'] - curr_position['y']) > .1 \
                    or abs(goal_position['z'] - curr_position['z']) > .1:
                curr_position = self.get_position()
                if default_timer() - movement_start > time_out:
                    break
            time.sleep(.5)  # allow some buffer in case robot is still doing final adjustments
        return response == 'ok'

    def get_speed(self) -> Dict:
        """
        get the current speed of the robot
        :return: dictionary where the keys are ['x', 'y', 'z', 'w1', 'w2', 'w3', 'w4'] and values are their speeds
        """
        self.pose_mutex.acquire()
        speed = self.speed.copy()
        self.pose_mutex.release()
        return speed

    def get_position(self) -> Dict:
        """
        get current position of robot relative to it's starting location
        :return: dictionary where the keys are ['x', 'y', 'z'] and values are how much the robot has moved in that
        direction
        """
        self.pose_mutex.acquire()
        position = self.position.copy()
        self.pose_mutex.release()
        return position

    def get_attitude(self) -> Dict:
        """
        get current attitude information
        :return: dictionary where the keys are ['pitch', 'roll', 'yaw'] and values are the current heading
        """
        self.pose_mutex.acquire()
        attitude = self.attitude.copy()
        self.pose_mutex.release()
        return attitude

    def __get_pose(self):
        while True:
            cmd = 'chassis position ?'
            response = super().send_command(cmd)
            self.pose_mutex.acquire()
            self.position = {k: float(v) for k, v in zip(self.position.keys(), response.split(' '))}
            self.pose_mutex.release()

            cmd = 'chassis speed ?'
            response = super().send_command(cmd)
            self.pose_mutex.acquire()
            self.speed = {k: float(v) for k, v in zip(self.speed.keys(), response.split(' '))}
            self.pose_mutex.release()

            cmd = 'chassis attitude ?'
            response = super().send_command(cmd)
            self.pose_mutex.acquire()
            self.attitude = {k: float(v) for k, v in zip(self.attitude.keys(), response.split(' '))}
            self.pose_mutex.release()
