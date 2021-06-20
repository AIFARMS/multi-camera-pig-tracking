"""
Simulates detection and tracking of pigs for a single camera
"""

import sys
sys.path.append('deepsort-tracking')

import cv2
import json
from collections import defaultdict
# from deep_sort.tracker import Tracker
# from deep_sort import nn_matching

# from tools import generate_detections as gdet
# from yolov4.annotate import Detector

import multiprocessing as multiproc
multiproc.set_start_method('fork')

class Camera(multiproc.context.Process):

    def __init__(self, stream, queue, track_prefix="", simulation_file=None):
        multiproc.context.Process.__init__(self)
        self.queue = queue
        self.track_prefix = track_prefix

        if simulation_file is None:
            # Definition of the parameters
            max_cosine_distance = 0.5

            # Initialize deepsort
            model_filename = 'networks//mars-small128.pb'
            encoder = gdet.create_box_encoder(model_filename, batch_size=1)
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, None)

            # Initialize Object Detector and Tracker
            video_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
            video_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            detector = Detector(video_width, video_height)

            self.tracker = Tracker(metric, detector, encoder)
            self.stream = stream
            self.run = self.track
        else:
            self.simulation_file = simulation_file
            self.run = self.simulate

    def track(self):
        frame_id = 0
        while True:
            ret, frame = self.stream.read()

            if not ret:
                break

            tracks = self.tracker.consume(frame)

            self.queue.put((frame_id, {"%s%d"%(self.track_prefix, t): bbox for t, bbox in tracks.items()}))
            frame_id += 1
        self.queue.put((-1, None))

    def get_tracks(self):
        return self.queue.get()

    def simulate(self):

        with open(self.simulation_file) as f:
            objects = json.load(f)["objects"]
        
        tracks_dict = defaultdict(dict)
        for id_dict in objects:
            for f in id_dict["frames"]:
                x, y = f["bbox"]["x"], f["bbox"]["y"]
                width, height = f["bbox"]["width"], f["bbox"]["height"]
                xmin, xmax = x-(width/2), x+(width/2) 
                ymin, ymax = y-(height/2), y+(height/2) 

                tracks_dict[f["frameNumber"]][id_dict["id"]] = [xmin, ymin, xmax, ymax]

        for frame_id, tracks in tracks_dict.items():
            self.queue.put((frame_id, {"%s%d"%(self.track_prefix, t): bbox for t, bbox in tracks.items()}))

        self.queue.put((-1, {}))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description = "Simulate using JSON file")
    parser.add_argument('--j', required=True, help="Json file which contains annotations")
    args = parser.parse_args()

    q = multiproc.Queue()
    c = Camera(None, q, simulation_file=args.j)

    c.start()

    while True:
        frame_id, tracks = q.get()
        if frame_id == -1:
            break

        print(frame_id, tracks)

    c.join()
