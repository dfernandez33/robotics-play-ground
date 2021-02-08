from Robomaster.controller import Controller
from imutils.video import FPS
import argparse
import imutils
import threading
from pynput import keyboard
from pynput.keyboard import KeyCode
import cv2
import numpy as np
import time


def track_object():
    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None
    fps = None
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = robomaster.camera.get_latest_frame()
        # check to see if we have reached the end of the stream
        if frame is None:
            continue
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame)
        (H, W) = frame.shape[:2]
        image_center = (W // 2, H // 2)
        cv2.circle(frame, image_center, 5, (0, 0, 255), 3)

        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                object_center = (x + w // 2, y + h // 2)
                cv2.circle(frame, object_center, 5, (0, 255, 0), 3)
                difference = image_center[0] - object_center[0]
                # (1 - gaussian(difference / (W / 2)))
                print('Width: {} Height: {}'.format(w, h))
                if difference < 0 and abs(difference) > 30:
                    robomaster.movement.drive(rotation_speed=15, duration=.05, await_response=False)
                elif difference > 0 and abs(difference) > 30:
                    robomaster.movement.drive(rotation_speed=-15, duration=.05, await_response=False)
            # update the FPS counter
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("c"):
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                                   showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()


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


def gaussian(x, mu=0.0, sig=.4):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


if __name__ == '__main__':

    robomaster = Controller('direct', enable_video=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tracker", type=str, default="kcf",
                    help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

    video_thread = threading.Thread(target=track_object, daemon=True)
    video_thread.start()
    listener = keyboard.Listener(
            on_press=on_press)
    listener.start()
    while True:
        print(robomaster.movement.position)
        time.sleep(1)

    robomaster.disable_video()
    cv2.destroyAllWindows()
