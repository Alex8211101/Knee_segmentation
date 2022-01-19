import glob
import os

import cv2
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))

input_dir = path+"/contours/"
output_dir = path+"/prepared/"

filesToRemove = glob.glob(output_dir+"*")
for f in filesToRemove:
    os.remove(f)

imagesToProcess = glob.glob(input_dir+"*")
for f in imagesToProcess:
    filename = f.split('\\')[1]
    img = cv2.imread(f, 0)
    img = img[:, :]
    pre = np.histogram(img, density=False)[0]
    img[np.bitwise_and(img > 0, img < 53)] = 1
    img[np.bitwise_and(img > 52, img < 104)] = 2
    img[np.bitwise_and(img > 103, img < 155)] = 3
    img[np.bitwise_and(img > 154, img < 206)] = 4
    img[img > 205] = 5
    cv2.imwrite(output_dir+filename, img)
    # logging if histogram is changed by swaping data
    after = np.histogram(img, density=False)[0]
    if pre != after:
        print("Filename: {filename}")
        print("Pre: "+pre)
        print("After: "+after)
        print("============================")
