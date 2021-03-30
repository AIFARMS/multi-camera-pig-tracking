import os
import cv2
import json
import argparse
from collections import defaultdict

from yolov4.annotate import Detector
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort import nn_matching
from absl import app, flags, logging
import time, random
import tensorflow as tf
from absl.flags import FLAGS
from deep_sort.detection import Detection
import matplotlib.pyplot as plt
import numpy as np
from deep_sort import preprocessing
from tqdm import tqdm

"""
Main file to run DeepSORT implementation with given tracker
"""
## Suppress Deprecated Warnings
import sys
if not sys.warnoptions:
	import warnings
	warnings.simplefilter("ignore")

mask_map = {
    "Ceiling_Cam"   : "ceiling",
    "Pen_B"         : "penb",
    "Pen_C"         : "penc"    
}

def annotate_video(video_path):
    video_prefix = video_path.split('.')[0]
    video_name = video_prefix.split('/')[-1]
    view = video_name.split('-')[-1]
    
    output_dict = {
        "videoFileName": video_name,
        "fullVideoFilePath": video_path,
        "stepSize": 0.1,
        "config": {
            "stepSize": 0.1,
            "playbackRate": 0.4,
            "imageMimeType": "image/jpeg",
            "imageExtension": ".jpg",
            "framesZipFilename": "extracted-frames.zip",
            "consoleLog": "0"
        },
        "objects":[]
    }

    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    #initialize deepsort
    model_filename = 'networks//mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    
    ## Initialize Object Detector and Tracker
    cap = cv2.VideoCapture(video_path)
    w, h, video_fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    det = Detector(w, h)
    deep_sort_tracker = Tracker(metric, det, encoder)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{video_prefix}-annotated.mp4", fourcc, video_fps, (w, h))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = 0
    pbar = tqdm(total=length, file=sys.stdout)

    mask_filter = cv2.imread(f"./yolov4/masks/{mask_map[view]}-maskfilter.png")
    if mask_filter.shape[:2] != (h, w):
    	mask_filter = cv2.resize(mask_filter, (w, h))
    
    id_frames = defaultdict(list)
    
    ## Read frames from stream
    while(True):
        ret, frame = cap.read()
    	
        if not ret: 
            break

    	## Tracker consumes a frame and spits out an annotated_frame
        
        if mask_filter is not None:
            frame = cv2.bitwise_and(frame, mask_filter)

        annotated_frame, bbox_dict = deep_sort_tracker.consume(frame)
    	
        
        ## Save the annotated frame
        #out.write(annotated_frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        
        pbar.update(1)

        for pig_id in bbox_dict:
            xmin, ymin, xmax, ymax = bbox_dict[pig_id]
            x = int((xmin+xmax)/2)
            y = int((ymin+ymax)/2)
            width, height = int(xmax-xmin), int(ymax-ymin)

            id_frames[pig_id].append({
                "frameNumber": total_frames,
                "bbox": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                },
                "isGroundTruth": "1",
                "visible": "1",
                "behaviour": "other"
            })

        total_frames += 1
        if total_frames == 100: break
    out.release()
    cap.release()
    #cv2.destroyAllWindows()
    del det, deep_sort_tracker 

    for pig_id, frames in id_frames.items():
        output_dict["objects"].append({
            "frames": frames,
            "id": pig_id
        })
    
    with open(f"{video_prefix}.json", "w") as f:
        json.dump(output_dict, f)

if __name__ == '__main__':
    
    videos_to_annotate = "Videos to annotate"
    for video_id in os.listdir(videos_to_annotate):
        print(f"Video ID: {video_id}")
        for video_name in os.listdir(os.path.join(videos_to_annotate, video_id)):
            annotate_video(os.path.join(videos_to_annotate, video_id, video_name))
