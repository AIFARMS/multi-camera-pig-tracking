import cv2
from collections import Counter, OrderedDict

class Pig:

    def __init__(self, pig_id, frame_id, boxes):
        self.id = pig_id
        self.boxes = OrderedDict({frame_id: boxes})

        self.activities = Counter()

    def __str__(self):
        return str(self.id)

    def update_boxes(self, frame_id, new_boxes):
        # print("Adding", new_boxes)
        if new_boxes is not None:
            self.boxes[frame_id] = new_boxes

    def get_boxes(self, frame_id):
        # print(self.current)
        return self.boxes[frame_id] if frame_id in self.boxes else (None, None)

    def increment_activity(self, activity):
        self.activities[activity] += 1

    def print(self):
        print(f'\033[32m  \n id: {self.id} | bb: {self.bounding_box} | ate: {self.ate} | drank: {self.drank} \033[0m')  # noqa: E501

ACTIVITY_COLOR = {
    'feeding' : (102, 3, 252)    
}

def annotate_frame(frame, pig_id, box, colors, activity):
    xmin, ymin, xmax, ymax = box
    try:
        color = colors[pig_id % len(colors)]
    except:
        color = colors[0]

    color = [i * 255 for i in color]
    if activity is None:
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 5)
    else:
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), ACTIVITY_COLOR[activity], 5)

    cv2.rectangle(frame, (int(xmin), int(ymin-60)), (int(xmin)+(3+len(str(pig_id)))*50, int(ymin)), color, -1)
    cv2.putText(frame, "Pig-" + str(pig_id),(int(xmin), int(ymin-10)),0, 2, (255,255,255),5)