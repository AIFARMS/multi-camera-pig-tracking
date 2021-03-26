import os
import pandas as pd
import cv2
import json
import numpy as np
from tqdm import tqdm

annotations_dir = 'merged.csv' #"/home/kotnana2/digital_ag/data/merged.csv"
base_dir = '.' #"/home/kotnana2/digital_ag/data"
out_dir = 'obj' #"/home/kotnana2/digital_ag/darknet/data/obj"

annotations = pd.read_csv(annotations_dir)

annotations_dict = {}
num_images = len(annotations.index)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

clahe = cv2.createCLAHE(clipLimit=5)
for x in tqdm(range(num_images)):
    image_path = annotations.iloc[x]['imagePath']
    temp = image_path.split("-")
    img_dir = temp[0]
    date = '-'.join(temp[1:4])
    timestamp = '-'.join(temp[4:6])
    img_path = os.path.join(base_dir, img_dir, date, timestamp, image_path)
    # print(os.path.exists(img_path))
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    
    out_image_path = image_path.split('.')[0] + '-night.jpg'
    out_img_path = os.path.join(out_dir, out_image_path)
    img = adjust_gamma(img, 1./3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = clahe.apply(img)
    cv2.imwrite(out_img_path, img)
    
    annots_list = json.loads(annotations.iloc[x]['Annotations'])
    out_annot_path = os.path.join(out_dir, image_path.split('.')[0] + "-night.txt")
    with open(out_annot_path, "w+") as f:
        for annot in annots_list:
            annot = json.loads(annot)
            x, y, w, h = annot["x"], annot["y"], annot["width"], annot["height"]
            x_cent = x + (w / 2)
            y_cent = y + (h / 2)
            x_cent_norm = x_cent / width
            y_cent_norm = y_cent / height
            width_norm = w / width
            height_norm = h / height
            f.write("0 {:0.5f} {:0.5f} {:0.5f} {:0.5f}\n".format(x_cent_norm, y_cent_norm, width_norm, height_norm)) 
    #print(out_img_path)
    #break
