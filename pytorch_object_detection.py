import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from Robomaster.controller import Controller
import cv2
from imutils.video import FPS
import torch.nn as nn
import timeit


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


if __name__ == '__main__':
    robomaster = Controller('direct')
    robomaster.start(enable_video=True)
    threshold = .85
    device = torch.device("cuda")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    fps = FPS().start()
    while True:
        start = timeit.default_timer()
        frame = robomaster.camera.get_latest_frame()
        if frame is None:
            continue
        print('Time to grab frame: {}'.format(timeit.default_timer() - start))
        (H, W) = frame.shape[:2]
        frame_tensor = transform(frame).to(device)
        start = timeit.default_timer()
        pred = model([frame_tensor])  # Pass the image to the model
        print('Time to run inference: {}'.format(timeit.default_timer() - start))
        start = timeit.default_timer()
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in
                      list(pred[0]['labels'].cpu().detach().numpy())]  # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
        pred_score = list(pred[0]['scores'].cpu().detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold]  # Get list of index with score greater than threshold.
        if len(pred_t) is not 0:
            pred_boxes = pred_boxes[:pred_t[-1] + 1]
            pred_class = pred_class[:pred_t[-1] + 1]
            for i in range(len(pred_boxes)):
                cv2.rectangle(frame, pred_boxes[i][0], pred_boxes[i][1], color=(0, 255, 0),
                              thickness=3)  # Draw Rectangle with the coordinates
                cv2.putText(frame, pred_class[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0),
                            thickness=3)  # Write the prediction class
        print('Time to process output: {}'.format(timeit.default_timer() - start))
        # update the FPS counter
        fps.update()
        fps.stop()
        # loop over the info tuples and draw them on our frame
        cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('prediction', frame)
        cv2.waitKey(1)

