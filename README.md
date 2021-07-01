# Multi Camera Pig Tracking
Official Implementation of **Tracking Grow-Finish Pigs Across Large Pens Using Multiple Cameras**

[CVPR2021 CV4Animals Workshop Poster](https://drive.google.com/file/d/1ecdUNkKhlcNxA0ZbvaZBc8qJdrLHAmUV/view)

<img src="data/multicam-tracking.gif" width="1000"></img>

## Dataset
The dataset can be found at [this](https://drive.google.com/drive/folders/1E2wW2aRENgy_TqlzfICn58ahbTHVIaK6?usp=sharing) link. 

The videos were acquired at the Imported Swine Research Lab (ISRL) at UIUC. The deployment video can be found [here](https://www.youtube.com/watch?v=DDw6cPtmHUA). It was annotated for grountruth global identities and bounding boxes using [this MATLAB Tool](https://github.com/AIFARMS/pig-annotation-tool/tree/master/Matlab%20Tools). 

## Files
[deepsort-tracking](deepsort-tracking): Contains code for detecting and tracking pigs using YOLOv4 and DeepSORT

[data/homography](data/homography): Contains pickled homography matrices for both the pens

[camera.py](camera.py): Detects and tracks pigs using the trained model in DeepSORT and YOLOv4

[manager.py](manager.py): Main file which uses the homography matrices to assign global identities


## Running the code
Download the dataset from [this](https://drive.google.com/drive/folders/1E2wW2aRENgy_TqlzfICn58ahbTHVIaK6?usp=sharing) link and place ``multicam-dataset`` folder in ``data/``. Note that we have already trained the model and extracted the output of DeepSORT into JSON files. **You can find the pretrained checkpoint [here](https://drive.google.com/file/d/1SCDtxM2WXQBMx1pqeoOg5JpYK3GvfDvx/view?usp=sharing)**

1. Run ``export DARKNET_PATH=./deepsort-tracking/yolov4/`` in terminal.
2. Run any one of the commands from [commands.txt](commands.txt), for instance: python3 manager.py --av data/multicam-dataset/0/0-Pen_B.mp4 --cv data/multicam-dataset/0/0-Ceiling_Cam.mp4 --cj data/multicam-dataset/0/0-Ceiling_Cam.json --aj data/multicam-dataset/0/0-Pen_B.json --cl 457

## Future Work

We are currently working on building action recognition models for pig behavior using ethograms. We can currently estimate the time spent by pigs near drinkers and feeders based on their proximity.

<img src="data/pigs.gif" width="1000"></img>

