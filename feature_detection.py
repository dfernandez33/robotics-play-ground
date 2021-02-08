import cv2
from Robomaster.controller import Controller
from pynput import keyboard
from pynput.keyboard import KeyCode
import numpy as np

def on_press(key):
    if key == KeyCode(char='w'):
        robomaster.movement.drive(x_speed=.5, duration=.5)
    elif key == KeyCode(char='s'):
        robomaster.movement.drive(x_speed=-.5, duration=.5)
    elif key == KeyCode(char='a'):
        robomaster.movement.drive(y_speed=-.5, duration=.5)
    elif key == KeyCode(char='d'):
        robomaster.movement.drive(y_speed=.5, duration=.5)
    elif key == KeyCode(char='e'):
        robomaster.movement.drive(rotation_speed=30, duration=.5)
    elif key == KeyCode(char='q'):
        robomaster.movement.drive(rotation_speed=-30, duration=.5)
    elif key == KeyCode(char='x'):
        return False

if __name__ == '__main__':
    robomaster = Controller('direct')
    robomaster.start(enable_video=True)

    fast = cv2.FastFeatureDetector_create()

    listener = keyboard.Listener(
        on_press=on_press
    )
    listener.start()
    while True:
        frame = robomaster.camera.get_latest_frame()
        # mask = np.zer
        if frame is None:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp = fast.detect(frame_gray, None)
        cv2.drawKeypoints(frame, kp, color=(255, 0, 0), outImage=frame)
        cv2.imshow('Features', frame)
        cv2.waitKey(1)
