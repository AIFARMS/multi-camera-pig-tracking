import os
import subprocess

videos_to_annotate = "Videos to annotate"
for video_id in os.listdir(videos_to_annotate):
    print(f"Video ID: {video_id}")
    for video_name in os.listdir(os.path.join(videos_to_annotate, video_id)):
        subprocess.run(["python3", "tracker_cv2.py", "-s", os.path.join(videos_to_annotate, video_id, video_name)])