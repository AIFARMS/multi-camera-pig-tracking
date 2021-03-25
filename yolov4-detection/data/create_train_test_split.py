import os
import random

images_dir = "obj"
split = 0.8

images = [x for x in os.listdir(images_dir) if "jpg" in x]
num_images = len(images)
random.shuffle(images)

with open("train.txt", "w+") as f:
    for x in range(0, int(num_images * split)):
        f.write(os.path.join("data", images_dir, images[x]) + "\n")

with open("test.txt", "w+") as f:
    for x in range(int(num_images * split), num_images):
        f.write(os.path.join("data", images_dir, images[x]) + "\n")
