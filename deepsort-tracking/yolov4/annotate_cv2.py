import cv2
import numpy as np
import argparse
import time
import os
from tqdm import tqdm
import pickle
import sys
from imutils.video import FPS

def parser():
	parser = argparse.ArgumentParser(description="YOLO Object Detection")
	parser.add_argument("--input", type=str, required=True,
						help="video source")
	parser.add_argument("--mask", type=str, default=None,
						help="mask")
	parser.add_argument("--weights", default="./backup/yolo-obj_last.weights",
						help="yolo weights path")
	parser.add_argument("--config_file", default="./cfg/yolo-obj.cfg",
						help="path to config file")
	parser.add_argument("--data_file", default="./data/obj.data",
						help="path to data file")
	parser.add_argument("--thresh", type=float, default=.25,
						help="remove detections with confidence below this value")
	parser.add_argument("--store", action='store_true',
						help="store detections in detections.pkl")
	parser.add_argument("--display", action='store_true',
						help="display detections")
	return parser.parse_args()

start_time = time.time()

args = parser()

classes = ["pig"]

# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Setup reading from video stream
video_path = args.input
video_stream = cv2.VideoCapture(video_path)
video_width  = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_stream.get(cv2.CAP_PROP_FPS))
length = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_name = os.path.split(video_path)[1].split('.')[0]
out = cv2.VideoWriter(video_name+'-annotated.mp4', fourcc, video_fps, (video_width, video_height))

if args.mask is not None:
	mask_filter = cv2.imread(args.mask)
	if mask_filter.shape[:2] != (video_height, video_width):
		mask_filter = cv2.resize(mask_filter, (video_width, video_height))

d_dict = {}
frame_id = 0
fps_counter = FPS().start()
pbar = tqdm(total=length, file=sys.stdout)
while True:
	ret, frame = video_stream.read()
	if frame is None: break
	height, width, _ = frame.shape

	prev_time = time.time()

	# Preprocess image
	if args.mask is not None:
		frame = cv2.bitwise_and(frame, mask_filter)
	blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

	# Forward pass
	net.setInput(blob)
	outputs = net.forward(ln)
	outputs = np.vstack(outputs)

	# Calculate bounding boxes
	boxes = []
	confidences = []
	classIDs = []
	for output in outputs:
		scores = output[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		if confidence > args.thresh:
			x, y, w, h = output[:4] * np.array([width, height, width, height])
			p0 = int(x - w//2), int(y - h//2)
			p1 = int(x + w//2), int(y + h//2)
			boxes.append([*p0, *p1])
			confidences.append(float(confidence))
			classIDs.append(classID)
	fps = int(1/(time.time() - prev_time))

	# Annotate
	d = []
	indices = cv2.dnn.NMSBoxes(boxes, confidences, args.thresh, args.thresh-0.1)
	if len(indices) > 0:
		for i in indices.flatten():
			x1, y1 = (boxes[i][0], boxes[i][1])
			x2, y2 = (boxes[i][2], boxes[i][3])
			d.append([(x1, y1), (x2, y2)])
			cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
			tx1, ty1, tx2, ty2 = x1, y1 - 5, x1 + 4 + 4., y1
			cv2.putText(frame, "{}: {:.2f}".format(classes[classIDs[i]], confidences[i]*100), (tx1, ty1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

	d_dict[frame_id] = d
	frame_id += 1
	out.write(frame)
	if args.display:
		cv2.imshow("YOLO pig detection", cv2.resize(frame, (960,540)))

		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	pbar.update(1)
	pbar.set_description("FPS: {}".format(fps))
	fps_counter.update()

if args.store:
	pickle.dump(d_dict, open(video_name + "-detections.pkl", "wb"))

video_stream.release()
out.release()

fps_counter.stop()
print("Elapsed time: {:.2f}".format(fps_counter.elapsed()))
print("Approximate FPS: {:.2f}".format(fps_counter.fps()))
print("Total time: {}".format(time.time() - start_time))
