1. clone yolo repository: http://github.com/alexeyab/darknet
2. replace existing data dir with data dir from this repo
3. run the `convert_pig_to_yolo.py` script to convert from our existing annotation format to yolo's format
4. run the `create_train_test_split.py` to split up into train/test sets
5. modify `Makefile` and set `GPU=1`, `CUDNN=1`, `LIBSO=1`, and uncomment `ARCH=` line corresponding to whatever GPU you have
6. run `make` in the current directory
7. to train, download existing [yolo weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137), and  run `./darknet detector train data/obj.data cfg/yolo-obj.cfg yolov4.conv.137`.
8. to annotate, run `python3 annotate.py --input INPUT_FILE`
