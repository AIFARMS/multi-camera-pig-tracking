from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
import pickle
from tqdm import tqdm
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
    return parser.parse_args()
 
start_time = time.time()

# Load model checkpoint that is to be evaluated
args = parser()
network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
network_width = darknet.network_width(network)
network_height = darknet.network_height(network)

## Setup reading from video stream
video_path = args.input
video_stream = cv2.VideoCapture(video_path)
video_width  = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_stream.get(cv2.CAP_PROP_FPS))
length = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

## Setup output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_name = video_path.split('/')[-1].split('.')[0]
out = cv2.VideoWriter(video_name+'-annotated.mp4', fourcc, video_fps, (video_width, video_height))

if args.mask is not None:
    mask_filter = cv2.imread(args.mask)
    if mask_filter.shape[:2] != (video_height, video_width):
        mask_filter = cv2.resize(mask_filter, (video_width, video_height))

d_dict = {}
frame_id = 0
fps_counter = FPS().start()
pbar = tqdm(total=length)
while True:
    ret, frame = video_stream.read()
    if frame is None: break
    height, width, _ = frame.shape

    prev_time = time.time()
    
    if args.mask is not None:
        frame = cv2.bitwise_and(frame, mask_filter)

    # Preprocess image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (network_width, network_height), interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(network_width, network_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

    # Forward pass
    detections = darknet.detect_image(network, class_names, img_for_detect, thresh=args.thresh)

    fps = int(1/(time.time() - prev_time))
    # print("FPS: {}".format(fps), end="\r", flush=True)

    d = []
   
    # Annotate
    for spec in detections:
        class_name, conf, bbox = spec
        x, y, w, h = bbox
        x1 = int(((x - (w / 2)) / network_width) * video_width)
        y1 = int(((y - (h / 2)) / network_height) * video_height)
        x2 = int(((x + (w / 2)) / network_width) * video_width)
        y2 = int(((y + (h / 2)) / network_height) * video_height)
        d.append([(x1, y1), (x2, y2)])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        tx1, ty1, tx2, ty2 = x1, y1 - 3, x1 + 4 + 4., y1
        cv2.putText(frame, "{}: {}".format(class_name, conf), (tx1, ty1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)   
 
    d_dict[frame_id] = d
    frame_id += 1
        
    # out.write(frame)
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
