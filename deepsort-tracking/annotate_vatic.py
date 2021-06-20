## Script for drawing bounding boxes on videos using vatic annotations

import os
import cv2
import sys
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description = "Annotate videos for vatic annotation format")
parser.add_argument('--stream_source', '-s', required=True, help="Source video stream. Default stream is the webcam")
parser.add_argument('--night', action='store_true')
args = parser.parse_args()

video_prefix = ''.join(args.stream_source.split('.')[:-1])

if not os.path.exists(f"{video_prefix}.json") or os.path.exists(f"{video_prefix}-annotated.mp4"):
	exit()

with open(f"{video_prefix}.json") as f:
	objects = json.load(f)["objects"]

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

## Infrared correction
CLAHE = cv2.createCLAHE(clipLimit=5)
GAMMA_TABLE = np.array([((i/255.0)**2.0)*255 for i in np.arange(0, 256)]).astype("uint8")

def infrared_correction(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	return  CLAHE.apply(cv2.LUT(frame, GAMMA_TABLE))

annotations = defaultdict(dict)
for id_dict in objects:
	for f in id_dict["frames"]:
		x, y = f["bbox"]["x"], f["bbox"]["y"]
		width, height = f["bbox"]["width"], f["bbox"]["height"]
		xmin, xmax = x-(width/2), x+(width/2) 
		ymin, ymax = y-(height/2), y+(height/2) 

		annotations[f["frameNumber"]][id_dict["id"]] = [xmin, ymin, xmax, ymax]

cap = cv2.VideoCapture(args.stream_source)
pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), file=sys.stdout)
w, h = 3*640, 3*360
out = cv2.VideoWriter(f"{video_prefix}-annotated.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (w,h))
frame_id = 0
while True:
	success, frame = cap.read()
	if not success:
		break

	# if args.night:
		# frame = infrared_correction(frame)

	for pig_id, bbox in annotations[frame_id].items(): 
		color = colors[int(pig_id) % len(colors)]
		color = [i * 255 for i in color]
		cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 4)
		cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-50)), (int(bbox[0])+(3+len(str(pig_id)))*30, int(bbox[1])), color, -1)
		cv2.putText(frame, f"Pig-{pig_id}",(int(bbox[0]), int(bbox[1]-10)),0, 1.25, (255,255,255),4)

	frame = cv2.resize(frame, (w,h))
	# cv2.imshow("Pigs", frame)
	out.write(frame)
	frame_id += 1
	pbar.update(1)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

out.release()
cap.release()
cv2.destroyAllWindows()

