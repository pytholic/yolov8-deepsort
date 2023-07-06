import cv2
import numpy as np
import sys
import glob
import time
import torch

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import plot_results

# from deep_sort.tracker import Tracker


class YoloDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.classes = self.model.names
        # print(self.classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device: ", self.device)

    def score_frame(self, frame):
        self.model.to(self.device)
        results = self.model(frame)
        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                label, coord = r[-1], r[0:4]
                detections.append((label, coord))

        return detections

    def plot_results(self, frame, detections):
        for label, coord in detections:
            x1, y1, x2, y2 = coord
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color=(255, 0, 0),
                thickness=4,
            )
            cv2.putText(
                frame,
                self.classes[int(label)],
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Results", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict(self, frame):
        detections = self.score_frame(frame)
        self.plot_results(frame, detections)


detector = YoloDetector()
img = cv2.imread("./test2.jpg")
detector.predict(img)
