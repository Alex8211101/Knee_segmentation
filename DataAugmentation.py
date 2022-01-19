import os
import random

import numpy as np
from numpy import expand_dims
from PIL import Image
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array, load_img)


def add_noise(tensor):
    noise = np.random.normal(0, 50*random.random(), tensor.shape)
    return np.clip(tensor+noise, 0., 255.)


x = 1
howMany = 5
path = os.path.dirname(os.path.abspath(__file__))
input_dir = path+"/png/"
masks_dir = path+"/prepared/"
output_dir = path+"/png2/"
output_masks_dir = path+"/prepared2/"
img_size = (256, 256)
shift = 0.2
# choose one of methods of generating additional images
imageDataGen = ImageDataGenerator()  # Z1 none
# imageDataGen = ImageDataGenerator(brightness_range=[0.5, 1.5]) #Z2
# imageDataGen = ImageDataGenerator(preprocessing_function=add_noise) #Z3
# imageDataGen = ImageDataGenerator(brightness_range=[0.5, 1.5], preprocessing_function=add_noise) #Z4


for i in range(1, 201):
    img = load_img(input_dir+f"{i}.png",
                   target_size=img_size, color_mode="grayscale")
    mask = Image.open(masks_dir+f"{i}.png")
    img = img_to_array(img)
    img = expand_dims(img, 0)
    iterator = imageDataGen.flow(img, batch_size=1)
    for j in range(howMany):
        generated = iterator.next()
        generatedImage = generated[0].astype(np.uint8).reshape(256, 256)
        Image.fromarray(generatedImage).save(output_dir+f'{x}.png')
        mask.save(output_masks_dir+f'{x}.png')
        x += 1
