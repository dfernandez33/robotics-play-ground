import socket
from threading import Lock
from Robomaster.basecommandmodule import BaseCommandModule


class LED(BaseCommandModule):
    # LED configurations
    LED_ALL = 'bottom_all'
    LED_FRONT = 'bottom_front'
    LED_BACK = 'bottom_back'
    LED_LEFT = 'bottom_left'
    LED_RIGHT = 'bottom_right'

    # LED effects
    LED_EFFECT_SOLID = 'solid'
    LED_EFFECT_OFF = 'off'
    LED_EFFECT_PULSE = 'pulse'
    LED_EFFECT_BLINK = 'blink'

    def __init__(self, control_socket: socket.socket, socket_lock: Lock):
        super(LED, self).__init__(control_socket, socket_lock)

    def set_LED(self, configuration: str, effect: str, red=0, green=0, blue=0, await_response=True) -> bool:
        cmd = 'led control comp {} r {} g {} b {} effect {}'.format(configuration, red, green, blue, effect)
        response = super().send_command(cmd, await_response)
        return response == 'ok'
