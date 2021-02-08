from Robomaster.movement import Movement
from Robomaster.sensor import Sensors
from Robomaster.camera import Camera
from Robomaster.gripper import Gripper
from Robomaster.led import LED
import socket
from threading import Lock


class Controller:
    video_port = 40921
    audio_port = 40922
    control_port = 40923
    message_push_port = 40924
    event_port = 40925
    ip_addresses = {'direct': '192.168.2.1', 'USB': '192.168.42.2'}

    def __init__(self, connection_mode='direct', enable_video=False):
        # Initialize connection
        self.ip = self.ip_addresses[connection_mode]
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Connecting to robot control port')
        self.control_socket.connect((self.ip, self.control_port))
        print('Connected')
        self.control_socket.send('command;'.encode('utf-8'))
        self.control_socket.recv(1024).decode('utf-8')  # block until response arrives

        # Initialize control modules
        self.socket_lock = Lock()
        self.movement = Movement(self.control_socket, self.socket_lock)
        self.sensors = Sensors(self.control_socket, self.socket_lock)
        self.camera = Camera(self.ip, self.video_port)
        self.LED = LED(self.control_socket, self.socket_lock)
        self.gripper = Gripper(self.control_socket, self.socket_lock)

        # Initialize control module states
        self.gripper.close()  # ensure gripper is closed on start up
        self.LED.set_LED(self.LED.LED_ALL, self.LED.LED_EFFECT_SOLID, red=255)
        if enable_video:
            self.enable_video()
            print('video enabled')
            self.camera.start()
            print('camera started')

    def enable_video(self):
        self.control_socket.send('stream on;'.encode('utf-8'))
        self.control_socket.recv(1024).decode('utf-8')  # block until response arrives

    def disable_video(self):
        self.control_socket.send('stream off;'.encode('utf-8'))
        self.control_socket.recv(1024).decode('utf-8')  # block until response arrives
