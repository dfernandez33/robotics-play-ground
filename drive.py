from Robomaster.controller import Controller
import threading
import cv2
import time


def monitor_ToF(controller: Controller):
    controller.sensors.toggle_ToF()
    while True:
        ir_distance = controller.sensors.get_ToF_reading(1)
        if ir_distance <= 10:
            controller.movement.stop()


if __name__ == '__main__':
    robomaster = Controller('direct')
    robomaster.start(enable_video=True)
    tof_thread = threading.Thread(target=monitor_ToF, args=(robomaster,), daemon=True)
    tof_thread.start()
    time.sleep(1)
    robomaster.camera.show_video()
    counter = 0
    frame = robomaster.camera.get_latest_frame()
    while True:
        if counter % 2 == 0:
            robomaster.movement.drive(x_speed=1, duration=1)
        else:
            robomaster.movement.drive(rotation_speed=90, duration=1)
        counter += 1
