"""
Main file used to generate Global IDs using multiple camera views
"""

from camera import Camera
from shapely.geometry import Polygon, box
import numpy as np
import pickle
import json
import multiprocessing as multiproc
# multiproc.set_start_method('fork') ## Context already set in camera.py

from disjoint_set import DisjointSet
from pigutils import annotate_frame
from collections import defaultdict, Counter

DEBUG = False

class CameraManager(multiproc.context.Process):

    def __init__(self, angled_camera, ceil_camera, queue, total_pigs, pen_name, ceil_lag):
        multiproc.context.Process.__init__(self)

        assert isinstance(angled_camera, Camera) and isinstance(ceil_camera, Camera)

        self.ceil_camera = ceil_camera
        self.angled_camera = angled_camera

        self.local_buffer = {
            'ceiling': {}, 'angled': {}
        }

        self.warp_dict = {}
        with open(f"data/homography/matrices-{pen_name}.pickle", "rb") as f:
            self.H = pickle.load(f)
            self.warp_dict['angled'] = pickle.load(f)
            self.warp_dict['ceiling'] = pickle.load(f)

        self.queue = queue
        self.union_find = DisjointSet()
        self.mapping_counter = Counter()

        self.total_pigs = total_pigs
        self.pigs_seen = 0
        self.pen_name = pen_name
        self.ceil_lag = ceil_lag

    def get_global_ids(self):
        return self.queue.get()

    def ceiling_filter(self, ceil_tracks):
        pop_list = []
        for ceil_id in ceil_tracks:
            xmin, _, xmax, _ = ceil_tracks[ceil_id]
            
            box_center = (xmin + xmax)/2
            if (box_center > 2010 and self.pen_name == "C") or (box_center <=2010 and self.pen_name == "B"):
                pop_list.append(ceil_id)

        for _id in pop_list:
            ceil_tracks.pop(_id, None)

        return ceil_tracks

    def run(self):
        ceil_done, angled_done = False, False
        while not ceil_done and not angled_done:

            fid, tracks = self.ceil_camera.get_tracks()
            ceil_done = ceil_done or fid == -1
            if not ceil_done:
                self.local_buffer['ceiling'][fid+self.ceil_lag] = tracks

            fid, tracks = self.angled_camera.get_tracks()
            angled_done = angled_done or fid == -1
            if not angled_done:
                self.local_buffer['angled'][fid] = tracks

            self.match_tracks_from_buffer()

        with open("mapping.json", "w") as f:
            json.dump(dict(self.mapping_counter), f)
        print(self.mapping_counter)
        print("Mapping Written")
        self.queue.put((-1, None))

    def match_tracks_from_buffer(self):
        matching_fids = []
        for fid in set(self.local_buffer['angled']).intersection(set(self.local_buffer['ceiling'])):
            matching_fids.append(fid)
        
        for fid in matching_fids:
            global_position_dict = self.match_tracks(self.ceiling_filter(self.local_buffer['ceiling'].pop(fid, None)), 
                                                    self.local_buffer['angled'].pop(fid, None))
            self.queue.put((fid, global_position_dict))
            
    def match_tracks(self, ceil_tracks, angled_tracks):
        top_angled_to_angled = {}
        for ceil_id in ceil_tracks:

            ## Ceil tracking ID has not been added to the disjoin set
            # if self.union_find.find(ceil_id) == ceil_id:
            ceil_to_topceil = self.transform_polygon(self.warp_dict['ceiling'], self.rectangle_to_polygon(ceil_tracks[ceil_id]))
            top_ceil_to_top_angled = self.transform_polygon(np.linalg.inv(self.H), self.scale_polygon(ceil_to_topceil))
            top_angled_to_angled[ceil_id] = self.transform_polygon(np.linalg.inv(self.warp_dict['angled']), top_ceil_to_top_angled)

        angled_to_ceil, ceil_to_angled = self.generate_global_id(angled_tracks, top_angled_to_angled)

        global_position_dict = {}

        ## Assign Pigs in the Angled view a global ID first (Cropped version to remove weak detections)
        for angled_id in set(angled_tracks.keys()) - set(angled_to_ceil.keys()):

            if not angled_id in self.union_find:
                self.union_find.union(angled_id, self.pigs_seen)
                self.pigs_seen += 1

            global_id = self.union_find.find(angled_id)
            global_position_dict[global_id] = (angled_tracks[angled_id], None)

        ## Assign Pigs in the Ceil view a global ID in the end (Cropped version??)
        for ceil_id in set(ceil_tracks.keys()) - set(ceil_to_angled.keys()):

            if not ceil_id in self.union_find:
                self.union_find.union(ceil_id, self.pigs_seen)
                self.pigs_seen += 1

            global_id = self.union_find.find(ceil_id)
            global_position_dict[global_id] = (None, ceil_tracks[ceil_id])

        ## Assign Pigs tracked by Homography a Global ID
        for ceil_id, angled_id in ceil_to_angled.items():

            self.mapping_counter[f"{ceil_id}-{angled_id}"] += 1

            if angled_id in self.union_find:

                if ceil_id in self.union_find:
                    ## Both local IDs already exist and have a Global ID 
                    ## BUT, this can lead to ID merges. Two or more pigs can get assigned to the same ID
                    ## So let's assign a completely new global ID now (Can be improved)

                    # common_id = min(self.union_find.find(ceil_id), self.union_find.find(angled_id))
                    # self.union_find.union(ceil_id, common_id)
                    # self.union_find.union(angled_id, common_id)

                    if self.union_find.find(ceil_id) != self.union_find.find(angled_id):

                        ## TODO: Create custom reset function
                        min_id, max_id = sorted((self.union_find.find(ceil_id), self.union_find.find(angled_id)))
                        if min_id not in global_position_dict:
                            self.union_find.reset(ceil_id, min_id)
                            self.union_find.reset(angled_id, min_id)
                        else:
                            self.union_find.reset(ceil_id, max_id)
                            self.union_find.reset(angled_id, max_id)
                else:
                    ## Assign the Global ID of angled_id to ceil_id
                    self.union_find.union(ceil_id, self.union_find.find(angled_id))

            else:
                if ceil_id in self.union_find:
                    ## Assign the Global ID of ceil_id to angled_id
                    self.union_find.union(angled_id, self.union_find.find(ceil_id))
                else:
                    ## Assign a unique ID to both of them
                    self.union_find.union(ceil_id, self.pigs_seen)
                    self.union_find.union(angled_id, self.pigs_seen)

                    self.pigs_seen += 1

            global_id = self.union_find.find(angled_id)
            global_position_dict[global_id] = (angled_tracks[angled_id], ceil_tracks[ceil_id])


        if DEBUG:
            print(list(self.union_find.itersets()))
            print(global_position_dict)

        return global_position_dict

    def generate_global_id(self, angled_tracks, top_angled_to_angled):
        if len(top_angled_to_angled) == 0:
            return {}, {}

        c2idx, idx2c, a2idx, idx2a = {}, {}, {}, {}
        overlap_matrix = []

        original_boxes = []
        for pig_id in angled_tracks:
            a2idx[pig_id] = len(original_boxes)
            idx2a[len(original_boxes)] = pig_id     
            original_boxes.append(box(*angled_tracks[pig_id]))
            
        for pig_id in top_angled_to_angled:
            c2idx[pig_id] = len(overlap_matrix)
            idx2c[len(overlap_matrix)] = pig_id
            
            poly = Polygon(list(top_angled_to_angled[pig_id].values()))
            overlap_matrix.append([o.intersection(poly).area for o in original_boxes])

        matrix = np.array(overlap_matrix)
        mapped_transformations = set()
        angled_to_ceil, ceil_to_angled = {}, {}
        while True:
            ## Get maximum
            c, a = np.unravel_index(matrix.argmax(), matrix.shape)
            
            ## If we have matched all transformations, then exit
            if matrix[c,a] == 0:
                # print("Matching done")
                break
            
            ## Set max value to 0 so that we don'c come across this again
            matrix[:,a] = 0
            
            ## Check if we have already mapped c with a better ID
            if c in mapped_transformations:
                continue
            mapped_transformations.add(c)
            
            if DEBUG:
                print("Angled ID: %s associated with Ceiling ID: %s"%(idx2a[a], idx2c[c]))
            angled_to_ceil[idx2a[a]] = idx2c[c]
            ceil_to_angled[idx2c[c]] = idx2a[a]

        return angled_to_ceil, ceil_to_angled

    @staticmethod
    def transform_polygon(H, poly_dict):
        """
        H : 3x3 matrix
        poly_dict : {"tl": [x1, y1], ...}
        
        new_poly_dict : {"tl": [H(x1, y1)], ...}
        """
        new_poly_dict = {}
        for pos, old_p in poly_dict.items():
            new_p = np.matmul(H, np.array(old_p + [1]))
            new_poly_dict[pos] = [int(new_p[0]/new_p[2]), int(new_p[1]/new_p[2])]
        return new_poly_dict

    @staticmethod
    def rectangle_to_polygon(corners):
        xmin, ymin, xmax, ymax = corners
        poly_dict = {
            "tl" : [xmin, ymin],
            "tr" : [xmax, ymin],
            "br" : [xmax, ymax],
            "bl" : [xmin, ymax]
        }
        return poly_dict

    def scale_polygon(self, poly_dict, ceil_to_pen=True):

        if self.pen_name == "C":
            if ceil_to_pen:
                return {pos: [int(x*1443/1578), int(y*578/1712)] for pos, [x, y] in poly_dict.items()}
            else:
                return {pos: [int(x*1578/1443), int(y*1712/578)] for pos, [x, y] in poly_dict.items()}

        if self.pen_name == "B":
            if ceil_to_pen:
                return {pos: [int(x*1545/1352), int(y*645/1317)] for pos, [x, y] in poly_dict.items()}
            else:
                return {pos: [int(x*1352/1545), int(y*1317/645)] for pos, [x, y] in poly_dict.items()}

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description = "Annotate videos based on Multi-Camera annotations")
    parser.add_argument('--av', required=True, help="Angled video stream")
    parser.add_argument('--aj', required=True, help="Angled video json")
    parser.add_argument('--cv', required=True, help="Ceiling video stream")
    parser.add_argument('--cj', required=True, help="Ceiling video json")
    parser.add_argument('--cl', required=True, type=int, help="Ceiling Lag (in terms of frames)")
    args = parser.parse_args()

    ## Initialize Camera
    aq, cq = multiproc.Queue(), multiproc.Queue()
    angled_camera = Camera(None, aq, track_prefix="a", simulation_file=args.aj)
    ceiling_camera = Camera(None, cq, track_prefix="c", simulation_file=args.cj)
    angled_camera.start(); ceiling_camera.start();

    ## Initialize Camera Manager
    PEN_NAME = "B" if "B" in args.aj else "C"
    TOTAL_PIGS = 17 if PEN_NAME == "B" else 16
    cmq = multiproc.Queue()
    camera_manager = CameraManager(angled_camera, ceiling_camera, cmq, total_pigs=TOTAL_PIGS, pen_name=PEN_NAME, ceil_lag=args.cl)
    camera_manager.start()

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, TOTAL_PIGS)]

    f = 3
    w, h = int(f*1280), int(f*360)
    print(w,h)
    angled_cap = cv2.VideoCapture(args.av)
    ceiling_cap = cv2.VideoCapture(args.cv)
    out = cv2.VideoWriter('multi-tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))

    ## Remove initial 30 frames to match offset
    for i in range(args.cl):
        angled_cap.read()

    angled_output_dict = {
        "videoFileName": args.av,
        "fullVideoFilePath": args.av,
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

    ceiling_output_dict = {
        "videoFileName": args.cv,
        "fullVideoFilePath": args.cv,
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

    angled_id_frames, ceiling_id_frames = defaultdict(list), defaultdict(list)

    while True:
        frame_id, global_position_dict = camera_manager.get_global_ids()

        ret1, angled_frame = angled_cap.read()
        ret2, ceil_frame = ceiling_cap.read()

        if frame_id == -1 or not ret1 or not ret2:
            break
        
        for pig_id in global_position_dict:
            angled_box, ceil_box = global_position_dict[pig_id]

            if angled_box is not None:
                annotate_frame(angled_frame, pig_id, angled_box, colors, activity=None)
                xmin, ymin, xmax, ymax = angled_box
                x = int((xmin+xmax)/2)
                y = int((ymin+ymax)/2)
                width, height = int(xmax-xmin), int(ymax-ymin)

                angled_id_frames[pig_id].append({
                    "frameNumber": frame_id,
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

            if ceil_box is not None:
                annotate_frame(ceil_frame, pig_id, ceil_box, colors, activity=None)
                xmin, ymin, xmax, ymax = ceil_box
                x = int((xmin+xmax)/2)
                y = int((ymin+ymax)/2)
                width, height = int(xmax-xmin), int(ymax-ymin)

                ceiling_id_frames[pig_id].append({
                    "frameNumber": frame_id-args.cl,
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

        # cv2.putText(angled_frame, "Frame ID: %d"%frame_id, (20, 100), 0, 3, (0,0,255), 10)
        stacked_frames = cv2.resize(np.hstack((ceil_frame, angled_frame)), (w, h))
        cv2.imshow("Multi Camera Tracking", stacked_frames)
        out.write(stacked_frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for pig_id, frames in angled_id_frames.items():
        angled_output_dict["objects"].append({
            "frames": frames,
            "id": pig_id
        })

    for pig_id, frames in ceiling_id_frames.items():
        ceiling_output_dict["objects"].append({
            "frames": frames,
            "id": pig_id
        })

    with open("angled.json", "w") as f:
        json.dump(angled_output_dict, f)

    with open("ceiling.json", "w") as f:
        json.dump(ceiling_output_dict, f)

    print("Done")

    angled_camera.join()
    ceiling_camera.join()
    camera_manager.join() 



