"""
Activity tracker for Pens B and C
Uses the global IDs from the camera manager 
"""

import os
import json
import pickle
import numpy as np
from shapely.geometry import Polygon, box as PolyBox
import multiprocessing as multiproc
from collections import defaultdict
from copy import deepcopy

from pigutils import Pig, annotate_frame
from camera import Camera
from manager import CameraManager

from intervaltree import IntervalTree

class ActivityTracker:

    def __init__(self, pen_name, base_timestamp):
        self.interval_index = IntervalTree()
        self.pen_name = pen_name
        self.base_timestamp = base_timestamp

        self.current_activities = {
            'feeding': {}, 
            'drinking': {}
        }

    def update_activity(self, frame_id, activity_dict):

        for activity, ids in activity_dict.items():

            ## Iterate through all the pigs in the activity dict and 
            ## set the starting frame for the id if the activity is not tracked
            for pig_id in ids:
                if pig_id not in self.current_activities[activity]:
                    self.current_activities[activity][pig_id] = frame_id

            ## For those IDs which were not seen in the current activity dict, 
            ## Complete and add their acticity in the interval_index
            self.add_activity(set(self.current_activities[activity].keys())-ids, activity, frame_id)

    def add_activity(self, completed_ids, activity, end_frame_id):

        for pig_id in completed_ids:
            start_frame_id = self.current_activities[activity].pop(pig_id, None)
            self.interval_index[start_frame_id: end_frame_id] = (activity, pig_id)

    def export_tracker(self, pigs, frame_id):

        ## Firstly, add all current activities in the interval index
        for activity in self.current_activities.copy():
            self.add_activity(self.current_activities[activity].copy(), activity, frame_id)

        base_dir = 'data/indices/'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        with open(os.path.join(base_dir, "Pen%s-%s.pkl" % (self.pen_name, self.base_timestamp)), "wb") as f:
            pickle.dump(self.interval_index, f)
            pickle.dump(pigs, f)

    def import_tracker(self, filename):
        with open(filename, "rb") as f:
            self.interval_index = pickle.load(f)
            self.pigs = pickle.load(f)

    def query(self, q_activity, start_frame, end_frame):
        activities = [a.data for a in self.interval_index.overlap(start_frame, end_frame)]

        return [(activity, pig_id) for activity, pig_id in activities if activity == q_activity]

class Pen(multiproc.context.Process):

    def __init__(self, pen_name, camera_manager, base_timestamp="", vis_q=None):
        multiproc.context.Process.__init__(self)

        assert pen_name in ["B", "C"]

        self.activity_tracker = ActivityTracker(pen_name, base_timestamp)
        with open(f"config/{pen_name}.json") as f:
            roi_dict = json.load(f)

        self.activity_params = {
            'feeding' : {
                'roi' : roi_dict["feeding_roi"],
                'poly_roi' : Polygon(roi_dict["feeding_roi"]),
                'overlap_threshold' : 0.1,
                'orientation_threshold' : 1.1
            },
            'drinking' : {
                'roi' : roi_dict["drinking_roi"],
                'poly_roi' : Polygon(roi_dict["drinking_roi"]),
                'overlap_threshold' : 0.1,
                'orientation_threshold' : 1.1
            }
        }
        self.pigs = {}
        self.camera_manager = camera_manager

        self.vis_q = vis_q

    @staticmethod
    def get_activity(pig_id, activity_dict):
        for activity, ids in activity_dict.items():
            if pig_id in ids:
                return activity
        return None

    def run(self):
        while True:
            frame_id, global_ids = self.camera_manager.get_global_ids()
            if frame_id == -1:
                break
        #     self.update_pigs(frame_id, global_ids)
        #     activity_dict = self.detect_activities(frame_id)

        #     self.activity_tracker.update_activity(frame_id, activity_dict)

        #     if self.vis_q is not None:
        #         ## Augment Global IDs with activity detected
        #         for pig_id, (angled_box, ceil_box) in global_ids.items():
        #             global_ids[pig_id] = (angled_box, ceil_box, self.get_activity(pig_id, activity_dict))

        #         self.vis_q.put((frame_id, global_ids, activity_dict))

        # if self.vis_q is not None:
        #     self.vis_q.put((-1, None, None))

        # self.activity_tracker.export_tracker(self.pigs, frame_id)

    def update_pigs(self, frame_id, global_ids):
        global_ids_copy = deepcopy(global_ids)
        for pig_id in self.pigs:
            self.pigs[pig_id].update_boxes(frame_id, global_ids_copy.pop(pig_id, None))

        for pig_id, boxes in global_ids_copy.items():
            self.pigs[pig_id] = Pig(pig_id, frame_id, boxes)

    def get_intersection(self, box, activity):
        """ 
        Get intersection of Pig bounding box with the Region of interest
        """
        if box is None:
            return 0, 0
        x_min, y_min, x_max, y_max = box
        box_poly = PolyBox(x_min, y_min, x_max, y_max)

        intersection_area = self.activity_params[activity]["poly_roi"].intersection(box_poly).area

        return intersection_area, box_poly.area

    def check_orientation(self, box, activity):
        xmin, ymin, xmax, ymax = box
        w = xmax-xmin
        h = ymax-ymin
        aspect_ratio = float(w/h)
        return aspect_ratio > self.activity_params[activity]['orientation_threshold']

    def detect_activities(self, frame_id):
        """
        Check the activity of Pigs by finding the intersection of the bbox
        with the Region of Interest
        """
        
        activity_dict = defaultdict(set)
        
        for pig_id, pig in self.pigs.items():
            latest_boxes = pig.get_boxes(frame_id)
            for box_id, activity in enumerate(["feeding", "drinking"]):
                
                box = latest_boxes[box_id] 
                intersection_area, box_area = self.get_intersection(box, activity)
                if box_area and (intersection_area / box_area) > self.activity_params[activity]['overlap_threshold'] and self.check_orientation(box, activity):
                    pig.increment_activity(activity)
                    activity_dict[activity].add(pig_id)
                
                ## Skip drinking for now, remove once Okan completes the drinking part
                # break

        return activity_dict

    def draw_roi(self, frame, activity):
        """
        Draw the Region of Interest on the current frame
        """
        c = (102, 3, 252)
        for i in range(len(self.activity_params[activity]['roi'])):
            pt1 = self.activity_params[activity]['roi'][i%4]
            pt2 = self.activity_params[activity]['roi'][(i+1)%4]
            cv2.line(frame, tuple(pt1), tuple(pt2), c, 5)

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    
    PEN_NAME = "C"
    aq, cq = multiproc.Queue(), multiproc.Queue()
    angled_camera = Camera(None, aq, track_prefix="a", simulation_file=f"data/detections/pen{PEN_NAME.lower()}-day-output.pickle")
    ceiling_camera = Camera(None, cq, track_prefix="c", simulation_file="data/detections/ceil-day-output.pickle")
    angled_camera.start(); ceiling_camera.start();

    TOTAL_PIGS = 17 if PEN_NAME == "B" else 16
    cmq = multiproc.Queue()
    camera_manager = CameraManager(angled_camera, ceiling_camera, cmq, total_pigs=TOTAL_PIGS, pen_name=PEN_NAME)
    camera_manager.start()

    ## Queue for visualizing output
    vis_q = multiproc.Queue()
    pen_c = Pen(pen_name=PEN_NAME, camera_manager=camera_manager, vis_q=vis_q)
    pen_c.start()

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, TOTAL_PIGS)]

    f = 3
    w, h = int(f*1280), int(f*360)
    angled = cv2.VideoCapture(f"data/videos/pen{PEN_NAME.lower()}-day.mp4")
    ceil = cv2.VideoCapture("data/videos/ceiling-day.mp4")
    out = cv2.VideoWriter('multi-tracking-activity.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))

    ## Remove initial 30 frames to match offset
    for i in range(30):
        angled.read()

    while True:
        frame_id, global_activity_dict, activity_dict = vis_q.get()
        # print(global_activity_dict)
        ret1, angled_frame = angled.read()
        ret2, ceil_frame = ceil.read()

        if frame_id == -1 or not ret1 or not ret2:
            break
        
        print(frame_id, global_activity_dict, activity_dict)
        for pig_id in global_activity_dict:
            angled_box, ceil_box, activity = global_activity_dict[pig_id]

            if angled_box is not None:
                annotate_frame(angled_frame, pig_id, angled_box, colors, activity)
                
            if ceil_box is not None:
                annotate_frame(ceil_frame, pig_id, ceil_box, colors, activity)

        pen_c.draw_roi(angled_frame, "feeding")
        pen_c.draw_roi(ceil_frame, "drinking") 

        shiftx, shifty = 150, 20
        cv2.rectangle(angled_frame, (shiftx, shifty), (shiftx+650, shifty+120), (0, 0, 0), -1)
        cv2.putText(angled_frame, "Pigs at Feeder: %d" % len(activity_dict["feeding"]), (shiftx+50, shifty+80), 
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=5, color=(255, 255, 0))  

        stacked_frames = cv2.resize(np.hstack((ceil_frame, angled_frame)), (w, h))
        cv2.imshow("Multi Camera Tracking", stacked_frames)
        out.write(stacked_frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    angled_camera.join()
    ceiling_camera.join()
    camera_manager.join() 
    pen_c.join()
