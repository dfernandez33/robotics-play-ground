import socket
from threading import Lock
from Robomaster.basecommandmodule import BaseCommandModule

class Sensors(BaseCommandModule):

    def __init__(self, control_socket: socket.socket, socket_lock: Lock):
        super(Sensors, self).__init__(control_socket, socket_lock)
        self.tof_on = False

    def toggle_ToF(self) -> bool:
        """
        toggles the state of all ToF sensors
        :return: whether the command was executed correctly
        """
        self.tof_on = not self.tof_on
        cmd = 'ir_distance_sensor measure {}'.format('on' if self.tof_on else 'off')
        response = super().send_command(cmd)
        return response == 'ok'

    def get_ToF_reading(self, id: int) -> float:
        """
        get the reading from the specified ToF sensor
        :param id: which ToF sensor to getting a reading from
        :return: the reading from the ToF sensor
        """
        if not self.tof_on:
            raise Exception('Time of flight sensor must be turned on before reading the value')

        cmd = 'ir_distance_sensor distance {} ?'.format(id)
        response = super().send_command(cmd)
        return float(response)
