import socket
from threading import Lock


class BaseCommandModule:

    def __init__(self, socket: socket.socket, socket_lock: Lock):
        self.socket = socket
        self.buffer_size = 1024
        self.socket_lock = socket_lock

    def send_command(self, command: str, await_respose=True) -> str:
        """
        Sends the command to the robot through the establish TCP connection
        :param command: the command to send to the robot
        :return: the string representation of the robot's response
        """
        command += ';'
        self.socket_lock.acquire()
        self.socket.send(command.encode('utf-8'))
        if await_respose:
            self.socket_lock.release()
            return self.socket.recv(self.buffer_size).decode('utf-8')
        self.socket_lock.release()
        return 'ok'  # if not waiting for repsonse assume it's ok
