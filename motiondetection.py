"""
Motion Detection Algorithm
By Okan Kocabalkanli
29th March 2021
Requires: skimage (pip install scikit-image)
"""
import cv2
import numpy as np
from skimage.measure import block_reduce


def show(img, name, r=True):
    if r:
        result = cv2.resize(img, (720, 480))
    else:
        result = img
    cv2.imshow(name, result)

"""
round_to_black_and_white()
@inputs: img: max value in numpy array
@output: return a black and white image
"""
def round_to_black_and_white(img, i=255):
    mask2 = img / i
    mask2 = (mask2 < 0.5).astype('uint8')
    mask2 = 1 - mask2
    return mask2 * i

"""
moving_pixel_counter(img):
@inputs: img
@outputs: number of white pixels
"""
def moving_pixel_counter(img):
    temp = (img / 255).astype('uint8')
    show(temp * 255, 'c', r=False)
    temp = temp.flatten()

    t1 = sum(temp) / len(temp)
    return t1

"""
similarity(prev, current, last_p=0, shrink=5):
@inputs: 
    prev: last frame
    current: current frame
    last_p: last similarity value
    shrink: shrink square side
@outputs: return the similarity value
"""
def similarity(prev, current, last_p=0, shrink=5):
    shrink_cons = (shrink, shrink)  # (int(y * shrink), int(x * shrink))
    # Convert images to Black and White
    rounded_prev = round_to_black_and_white(prev)
    rounded_current = round_to_black_and_white(current)
    #MaxPool the Black and White Images
    shrink_prev = block_reduce(rounded_prev, shrink_cons, np.max).astype('uint8')
    shrink_current = block_reduce(rounded_current, shrink_cons, np.max).astype('uint8')

    #Xor to mark changes as white pixels and same pixels as black
    c_xor_p = np.bitwise_xor(shrink_current, shrink_prev)

    #count white pixels
    t1 = moving_pixel_counter(c_xor_p)
    # check different in similarity since last no movement
    return abs(t1 - last_p) * (2 - 1 / (shrink * shrink))


if __name__ == "__main__":

    # HYPERPARAMS
    firstime = True
    n_frame = 0 #number of frames
    frame_checkpoint = 10 #number of frames between each similarity check
    last_similarity_value = 0 #last similarity value
    # VIDEO INPUT
    cap = cv2.VideoCapture('data\\videos\\ceiling-night.mp4')
    mask = cv2.imread("data\\masks\\ceiling_mask.png")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = np.bitwise_and(frame, mask)  # mask the frame
        if (firstime):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # create grayscale
            gray = cv2.blur(gray, (30, 30)) #blur the image to even out natural light

        if (not firstime and n_frame % frame_checkpoint == 0):
            similarity_value = similarity(last_frame, gray, last_similarity_value, shrink=15) #calculate similarity
            similarity_percentage = similarity_value*100
            if (similarity_percentage < 1):
                print("No movement", similarity_percentage)
                last_similarity_value = similarity_value
            else:
                print("Movement", similarity_percentage)

        show(frame, "frame")
        if (n_frame % frame_checkpoint == 0):
            last_frame = gray
            firstime = False
        n_frame += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
