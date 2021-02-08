import cv2
import threading
import time


class Camera:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.capture = None
        self.video_capture_thread = threading.Thread(target=self.__capture_video, daemon=True)
        self.show_video_flag = False
        self.stop_flag = True
        self.frame = None

    def start(self):
        self.capture = cv2.VideoCapture(f'tcp://{self.ip}:{self.port}')
        time.sleep(2.0)
        self.stop_flag = False
        self.video_capture_thread.start()

    def get_latest_frame(self):
        if self.stop_flag:
            raise Exception('Make sure you start video capture before attempting to get latest frame.')
        return self.frame

    def toggle_video(self):
        self.show_video_flag = not self.show_video_flag

    def stop(self):
        self.show_video_flag = False
        self.stop_flag = True
        self.capture.release()
        self.video_capture_thread.join()

    def __capture_video(self):
        while True:
            if self.stop_flag:
                break
            if self.capture.isOpened():
                status, self.frame = self.capture.read()
                if self.show_video_flag:
                    cv2.imshow('Robomaster Feed', self.frame)
                    cv2.waitKey(1)

