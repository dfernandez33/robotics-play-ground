import socket
from threading import Lock
from Robomaster.basecommandmodule import BaseCommandModule


class Gripper(BaseCommandModule):

    def __init__(self, control_socket: socket.socket, socket_lock: Lock):
        super(Gripper, self).__init__(control_socket, socket_lock)
        self.is_open = False

    def open(self, force=1) -> bool:
        if self.is_open:
            return True
        cmd = 'robotic_gripper open {}'.format(force)
        response = super().send_command(cmd)
        self.is_open = True
        return response == 'ok'

    def close(self, force=1) -> bool:
        if not self.is_open:
            return True
        cmd = 'robotic_gripper close {}'.format(force)
        response = super().send_command(cmd)
        self.is_open = False
        return response == 'ok'
