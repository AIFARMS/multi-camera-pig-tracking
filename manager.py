from camera import Camera
from shapely.geometry import Polygon, box
import numpy as np
import pickle
import multiprocessing as multiproc
# multiproc.set_start_method('fork') ## Context already set in camera.py

from disjoint_set import DisjointSet
from pigutils import annotate_frame

DEBUG = False

class CameraManager(multiproc.context.Process):

    def __init__(self, angled_camera, ceil_camera, queue, total_pigs, pen_name):
        multiproc.context.Process.__init__(self)

        assert isinstance(angled_camera, Camera) and isinstance(ceil_camera, Camera)

        self.ceil_camera = ceil_camera
        self.angled_camera = angled_camera

        self.local_buffer = {
            'ceil': {}, 'angled': {}
        }

        self.warp_dict = {}
        with open("data/homography/matrices.pickle", "rb") as f:
            self.H = pickle.load(f)
            self.warp_dict['penc'] = pickle.load(f)
            self.warp_dict['ceil'] = pickle.load(f)

        self.queue = queue
        self.union_find = DisjointSet()

        self.total_pigs = total_pigs
        self.pigs_seen = 0
        self.pen_name = pen_name
        if self.pen_name == "C":
            self.ceil_offset = 30
        else:
            self.ceil_offset = 0

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
        while not ceil_done or not angled_done:

            fid, tracks = self.ceil_camera.get_tracks()
            ceil_done = ceil_done and fid == -1
            if not ceil_done:
                self.local_buffer['ceil'][fid+self.ceil_offset] = tracks

            fid, tracks = self.angled_camera.get_tracks()
            angled_done = angled_done and fid == -1
            if not angled_done:
                self.local_buffer['angled'][fid] = tracks

            self.match_tracks_from_buffer()

        self.queue.put((-1, None))

    def match_tracks_from_buffer(self):
        matching_fids = []
        for fid in set(self.local_buffer['angled']).intersection(set(self.local_buffer['ceil'])):
            matching_fids.append(fid)
        
        for fid in matching_fids:
            global_position_dict = self.match_tracks(self.ceiling_filter(self.local_buffer['ceil'].pop(fid, None)), 
                                                    self.local_buffer['angled'].pop(fid, None))
            self.queue.put((fid, global_position_dict))
            
    def match_tracks(self, ceil_tracks, angled_tracks):
        top_angled_to_angled = {}
        for ceil_id in ceil_tracks:

            ## Ceil tracking ID has not been added to the disjoin set
            # if self.union_find.find(ceil_id) == ceil_id:
            ceil_to_topceil = self.transform_polygon(self.warp_dict['ceil'], self.rectangle_to_polygon(ceil_tracks[ceil_id]))
            top_ceil_to_top_angled = self.transform_polygon(np.linalg.inv(self.H), self.scale_polygon(ceil_to_topceil))
            top_angled_to_angled[ceil_id] = self.transform_polygon(np.linalg.inv(self.warp_dict['penc']), top_ceil_to_top_angled)

        angled_to_ceil, ceil_to_angled = self.generate_global_id(angled_tracks, top_angled_to_angled)

        global_position_dict = {}

        ## Assign Pigs in the Angled view a global ID first (Cropped version to remove weak detections)
        for angled_id in set(angled_tracks.keys()) - set(angled_to_ceil.keys()):

            if self.pigs_seen < self.total_pigs:
                self.union_find.union(angled_id, self.pigs_seen)
                self.pigs_seen += 1

            global_id = self.union_find.find(angled_id)
            global_position_dict[global_id] = (angled_tracks[angled_id], None)

        ## Assign Pigs tracked by Homography a Global ID
        for ceil_id, angled_id in ceil_to_angled.items():

            if self.pigs_seen < self.total_pigs:
                ## Assign the pig found in the ceil_id and angled_id a unique Global identifier
                self.union_find.union(ceil_id, self.pigs_seen)
                self.union_find.union(angled_id, self.pigs_seen)

                self.pigs_seen += 1

            else:
                ## Assign 
                self.union_find.union(ceil_id, angled_id)

            global_id = self.union_find.find(angled_id)
            global_position_dict[global_id] = (angled_tracks[angled_id], ceil_tracks[ceil_id])

        ## Assign Pigs in the Ceil view a global ID in the end (Cropped version??)
        for ceil_id in set(ceil_tracks.keys()) - set(ceil_to_angled.keys()):

            if self.pigs_seen < self.total_pigs:
                self.union_find.union(ceil_id, self.pigs_seen)
                self.pigs_seen += 1

            global_id = self.union_find.find(ceil_id)
            global_position_dict[global_id] = (None, ceil_tracks[ceil_id])

        if DEBUG:
            print(list(self.union_find.itersets()))
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

    @staticmethod
    def scale_polygon(poly_dict, ceil_to_pen=True):
        if ceil_to_pen:
            return {pos: [int(x*1443/1578), int(y*578/1712)] for pos, [x, y] in poly_dict.items()}
        else:
            return {pos: [int(x*1578/1443), int(y*1712/578)] for pos, [x, y] in poly_dict.items()}

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    aq, cq = multiproc.Queue(), multiproc.Queue()
    angled_camera = Camera(None, aq, track_prefix="a", simulation_file="data/detections/penc-day-output.pickle")
    ceiling_camera = Camera(None, cq, track_prefix="c", simulation_file="data/detections/ceil-day-output.pickle")
    angled_camera.start(); ceiling_camera.start();

    TOTAL_PIGS = 17
    cmq = multiproc.Queue()
    camera_manager = CameraManager(angled_camera, ceiling_camera, cmq, total_pigs=TOTAL_PIGS, pen_name="C")
    camera_manager.start()

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, TOTAL_PIGS)]

    f = 3
    w, h = int(f*1280), int(f*360)
    angled = cv2.VideoCapture("data/videos/penc-day.mp4")
    ceil = cv2.VideoCapture("data/videos/ceiling-day.mp4")
    out = cv2.VideoWriter('multi-tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))

    ## Remove initial 30 frames to match offset
    for i in range(30):
        angled.read()

    while True:
        frame_id, global_position_dict = camera_manager.get_global_ids()
        print(frame_id, global_position_dict)
        ret1, angled_frame = angled.read()
        ret2, ceil_frame = ceil.read()

        if frame_id == -1 or not ret1 or not ret2:
            break
        
        for pig_id in global_position_dict:
            angled_box, ceil_box = global_position_dict[pig_id]

            if angled_box is not None:
                annotate_frame(angled_frame, pig_id, angled_box, colors, activity=None)
                
            if ceil_box is not None:
                annotate_frame(ceil_frame, pig_id, ceil_box, colors, activity=None)

        cv2.putText(angled_frame, "Frame ID: %d"%frame_id, (20, 100), 0, 3, (0,0,255), 10)
        stacked_frames = cv2.resize(np.hstack((ceil_frame, angled_frame)), (w, h))
        cv2.imshow("Multi Camera Tracking", stacked_frames)
        out.write(stacked_frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    angled_camera.join()
    ceiling_camera.join()
    camera_manager.join() 



