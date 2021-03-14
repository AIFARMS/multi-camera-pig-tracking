import cv2
import numpy as np
import argparse
import time
import os
from tqdm import tqdm
import pickle
import sys
from imutils.video import FPS

class Detector(object):


	def __init__(self, net_width, net_height):
		self.mask = "./yolov4/masks/penb-maskfilter.png" #open mask
		self.weights = "./yolov4/backup/yolo-tiny.weights"
		self.config_file = "./yolov4/cfg/yolo-tiny-obj.cfg"
		self.data_file = "./yolov4/data/obj.data"
		self.thresh = .25

		self.video_width = net_width
		self.video_height = net_height
		self.net = cv2.dnn.readNetFromDarknet(self.config_file, self.weights)
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

		self.ln = self.net.getLayerNames()
		self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

		self.mask_filter = cv2.imread(self.mask)
		if self.mask_filter.shape[:2] != (self.video_height, self.video_width):
			self.mask_filter = cv2.resize(self.mask_filter, (self.video_width, self.video_height))

	def get_localization(self, frame):

		height, width, _ = frame.shape
		
		blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)


		self.net.setInput(blob)
		outputs = self.net.forward(self.ln)
		outputs = np.vstack(outputs)
       
		boxes = []
		confidences = []
		classIDs = []
		for output in outputs:
			scores = output[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > self.thresh:
				x, y, w, h = output[:4] * np.array([width, height, width, height])
				p0 = int(x - w//2), int(y - h//2)
				p1 = int(x + w//2), int(y + h//2)
				boxes.append([*p0, *p1])
				confidences.append(float(confidence))
				classIDs.append(classID)

		indices = cv2.dnn.NMSBoxes(boxes, confidences, self.thresh, self.thresh-0.1)

		filtered_boxes = []
		filtered_confidences = []
		filtered_classIDs = []

		if len(indices) > 0:
			for i in indices.flatten():
				x1, y1 = (boxes[i][0], boxes[i][1])
				x2, y2 = (boxes[i][2], boxes[i][3])
				width = x2 - x1
				height = y2 - y1
				filtered_boxes.append(np.array([int(x1), int(y1), int(width), int(height)]))
				filtered_confidences.append(confidences[i])
				filtered_classIDs.append("pig")
		
		return filtered_boxes, filtered_confidences, filtered_classIDs