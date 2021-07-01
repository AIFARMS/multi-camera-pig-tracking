from ctypes import *
import random
import os
import cv2
import time
import yolov4.darknet as darknet
import argparse
import numpy as np


class Detector(object):


    def __init__(self, net_width, net_height):
        self.mask = None
        self.weights = "./yolov4/backup/yolo.weights"
        self.config_file = "./yolov4/cfg/yolo-obj.cfg"
        self.data_file = "./yolov4/data/obj.data"
        self.thresh = .25
        self.network, self.class_names, self.class_colors = darknet.load_network(
            self.config_file,
            self.data_file,
            self.weights,
            batch_size=1
        )
        self.network_width = darknet.network_width(self.network)
        self.network_height = darknet.network_height(self.network)
        self.video_width = net_width
        self.video_height = net_height

    def get_localization(self, frame):
        
        height, width, _ = frame.shape
        if self.mask is not None:
            mask_filter = cv2.imread(self.mask)
            if mask_filter.shape[:2] != frame.shape[:2]:
                mask_filter = cv2.resize(mask_filter, (frame.shape[1], frame.shape[0]))
            frame = cv2.bitwise_and(frame, mask_filter)


        #Preprocess image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.network_width, self.network_height), interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(self.network_width, self.network_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

        # Forward pass
        detections = darknet.detect_image(self.network, self.class_names, img_for_detect, thresh=self.thresh)

        boxes = []
        names = []
        confidence = []
        for spec in detections:
            class_name, conf, bbox = spec
            x, y, w, h = bbox
            x1 = int(((x - (w / 2)) / self.network_width) * self.video_width)
            y1 = int(((y - (h / 2)) / self.network_height) * self.video_height)
            x2 = int(((x + (w / 2)) / self.network_width) * self.video_width)
            y2 = int(((y + (h / 2)) / self.network_height) * self.video_height)
            width = x2 - x1
            height = y2 - y1
            new_bbox = np.array([x1, y1, width, height])
            boxes.append(new_bbox)
            names.append(class_name)
            confidence.append(conf)

        return boxes, confidence, names
