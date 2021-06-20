## Script for drawing bounding boxes on videos using Multicamera annotations

import argparse
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict, defaultdict

parser = argparse.ArgumentParser(description = "Annotate videos based on Multi-Camera annotations")
parser.add_argument('--av', required=True, help="Angled video stream")
parser.add_argument('--aj', required=True, help="Angled video json")
parser.add_argument('--cv', required=True, help="Ceiling video stream")
parser.add_argument('--cj', required=True, help="Ceiling video json")
args = parser.parse_args()

with open(args.aj) as f:
	angled_annotations = json.loads(f.read())

with open(args.cj) as f:
	ceiling_annotations = json.loads(f.read())

angled_mapping = OrderedDict({'interpolated': {}})
for d in angled_annotations['Data']:
	angled_mapping[d['Frame_ID']] = d['Record']

ceiling_mapping = OrderedDict({'interpolated': {}})
for d in ceiling_annotations['Data']:
	ceiling_mapping[d['Frame_ID']] = d['Record']

angled_cap = cv2.VideoCapture(args.av)
ceiling_cap = cv2.VideoCapture(args.cv)
video_prefix = ''.join(args.av.split('.')[:-1])

## Empty out the initial frames until both videos sync
angled_fid = 1
while angled_fid < angled_annotations['FrameInfo']['FirstCordinationFrame']:
	angled_cap.read()
	angled_fid += 1

ceiling_fid = 1
while ceiling_fid < ceiling_annotations['FrameInfo']['FirstCordinationFrame']:
	ceiling_cap.read()
	ceiling_fid += 1

w, h = 2*1280, 2*360
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
out = cv2.VideoWriter(f"{video_prefix}-joint.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (w,h))

def get_record(frame_id, mapping):
	if frame_id in mapping:

		## Interpolate mapping
		future_frame_id = frame_id+15
		assert future_frame_id in mapping
		
		joint_mapping = defaultdict(dict)
		for d in mapping[frame_id]:
			joint_mapping[d['Animal_ID']]['prev'] = d['Cord']
		for d in mapping[future_frame_id]:
			joint_mapping[d['Animal_ID']]['next'] = d['Cord']

		for offset in range(1, 15):
			record = []
			for animal_id, d in joint_mapping.items():
				if 'prev' in d and 'next' in d:
					new_d = {'Animal_ID': animal_id}

					prev_x, prev_y, prev_w, prev_h = d['prev']
					next_x, next_y, next_w, next_h = d['next']

					new_x = prev_x + offset * (next_x-prev_x) / 15
					new_y = prev_y + offset * (next_y-prev_y) / 15
					new_w = prev_w + offset * (next_w-prev_w) / 15
					new_h = prev_h + offset * (next_h-prev_h) / 15

					new_d['Cord'] = [new_x, new_y, new_w, new_h]
					new_d['interpolated'] = True

					record.append(new_d)

			mapping['interpolated'][frame_id+offset] = record
		
		return mapping[frame_id]

	if frame_id in mapping['interpolated']:
		return mapping['interpolated'][frame_id]

	return []

def annotate_frame(frame, record):
	for d in record:
		pig_id = d['Animal_ID']
		bbox = [int(k) for k in d['Cord']] 

		color = colors[int(pig_id) % len(colors)]
		color = [i * 255 for i in color]
		_w, _h = bbox[2], bbox[3]
		xmin, ymin = bbox[0], bbox[1]
		xmax, ymax = xmin+_w, ymin+_h

		cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2 if 'interpolated' in d else 10)
		cv2.rectangle(frame, (xmin, ymin-30), (xmin+(3+len(str(pig_id)))*17, ymin), color, -1)
		cv2.putText(frame, f"Pig-{pig_id}",(xmin, ymin-10),0, 0.75, (255,255,255),2)

while True:
	ret1, angled_frame = angled_cap.read()
	ret2, ceiling_frame = ceiling_cap.read()

	if not ret1 or not ret2:
		break

	annotate_frame(angled_frame, get_record(angled_fid, angled_mapping))
	annotate_frame(ceiling_frame, get_record(ceiling_fid, ceiling_mapping))	

	frame = np.hstack((ceiling_frame, angled_frame))
	frame = cv2.resize(frame, (w,h))
	# cv2.imshow("Pigs", frame)
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break
		
	out.write(frame)
	angled_fid += 1
	ceiling_fid += 1

# out.release()
angled_cap.release()
ceiling_cap.release()
cv2.destroyAllWindows()
