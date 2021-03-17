import numpy as np
from shapely.geometry import Polygon, box as PolyBox
import multiprocessing as multiproc
from collections import defaultdict
from copy import deepcopy

from pigutils import Pig, annotate_frame
from camera import Camera
from manager import CameraManager

class Pen(multiproc.context.Process):

    def __init__(self, pen_name, camera_manager, vis_q=None):
        multiproc.context.Process.__init__(self)

        feeding_roi = [(2771, 539), (3203, 1315), (2963, 1640), (2663, 815)]
        ## Create a drinking ROI
        self.activity_params = {
            'feeding' : {
                'roi' : feeding_roi,
                'poly_roi' : Polygon(feeding_roi),
                'overlap_threshold' : 0.1,
                'orientation_threshold' : 1.1
            }
            ## Add params for drinking
        }
        self.pigs = {}
        self.camera_manager = camera_manager

        self.vis_q = vis_q

    def run(self):
        while True:
            frame_id, global_ids = self.camera_manager.get_global_ids()
            if frame_id == -1:
                break
            self.update_pigs(frame_id, global_ids)
            activity_dict = self.detect_activities(frame_id)

            if self.vis_q is not None:
                ## Augment Global IDs with activity detected
                for pig_id, (angled_box, ceil_box) in global_ids.items():
                    for activity, ids in activity_dict.items():
                        global_ids[pig_id] = (angled_box, ceil_box, activity if pig_id in ids else None)

                self.vis_q.put((frame_id, global_ids, activity_dict))

        if self.vis_q is not None:
            self.vis_q.put((-1, None, None))

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
                break

        return activity_dict

    def draw_roi(self, frame, activity):
        """Draw the Region of Interest on the current frame
        """
        c = (102, 3, 252)
        for i in range(1, len(self.activity_params[activity]['roi'])):
            pt1 = self.activity_params[activity]['roi'][i-1]
            pt2 = self.activity_params[activity]['roi'][i]
            cv2.line(frame, pt1, pt2, c, 5)
        cv2.line(frame, self.activity_params[activity]['roi'][0], self.activity_params[activity]['roi'][-1], c, 5)

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

    ## Queue for visualizing output
    vis_q = multiproc.Queue()
    pen_c = Pen(pen_name="C", camera_manager=camera_manager, vis_q=vis_q)
    pen_c.start()

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, TOTAL_PIGS)]

    f = 3
    w, h = int(f*1280), int(f*360)
    angled = cv2.VideoCapture("data/videos/penc-day.mp4")
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
        
        for pig_id in global_activity_dict:
            angled_box, ceil_box, activity = global_activity_dict[pig_id]

            if angled_box is not None:
                annotate_frame(angled_frame, pig_id, angled_box, colors, activity)
                
            if ceil_box is not None:
                annotate_frame(ceil_frame, pig_id, ceil_box, colors, activity)

        pen_c.draw_roi(angled_frame, "feeding")
        ## pen_c.draw_roi(ceil_frame, "drinking") Uncomment once drinking params are done

        shiftx, shifty = 150, 20
        cv2.rectangle(angled_frame, (shiftx, shifty), (shiftx+650, shifty+120), (0, 0, 0), -1)
        cv2.putText(angled_frame, "Pigs at Feeder: %d" % len(activity_dict["feeding"]), (shiftx+50, shifty+80), 
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=5, color=(255, 255, 0))  

        # cv2.putText(angled_frame, "Pigs at Feeder: %d"%len(activity_dict["feeding"]), (20, 100), 0, 2, (0,0,255), 10)
        stacked_frames = cv2.resize(np.hstack((ceil_frame, angled_frame)), (w, h))
        cv2.imshow("Multi Camera Tracking", stacked_frames)
        out.write(stacked_frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    angled_camera.join()
    ceiling_camera.join()
    camera_manager.join() 
    pen_c.join()
