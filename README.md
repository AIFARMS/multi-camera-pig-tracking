# Multi Camera Pig Tracking
Official Implementation of "Continuous Multi-Camera Pig Tracking to Estimate Drinking and Feeding Behavior in Commercial Grow-Finish Pens"

<img src="data/pigs.gif" width="1000"></img>

We have extracted the output of Deepsort and stored it in ``data/detections/``. To run our code: 

1. Download the videos from [here](https://drive.google.com/drive/folders/1oYSxkPNxPle8qn5sxGyNLpIp9mMomlpB?usp=sharing) and place them in the ``data/videos`` folder
2. Run ``export DARKNET_PATH=./yolov4-detection/`` in terminal.
3. Run ``python3 manager.py``

