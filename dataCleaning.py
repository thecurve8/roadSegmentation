# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:10:41 2020

@author: Alexander

This script is used to remove images from the train dataset.
Some images have big white chunks. 
The image and its label are removed if there are more than 30% white pixels.
Removed images and labels are put in new folders (removed_sat and removed_map)

This script can take a long time to run.

PIL and numpy are used.
"""

import os
from PIL import Image
import numpy as np
from settings import TRAIN_FOLDER


def whitePixelsPercentage(image):
    """Calculates percentage of white pixels in an RGB image
    
    Parameters
    ----------
    image : numpy array
        numpy array of the image of size (None, None, 3)
        
    Returns
    -------
    white/total: float
        percentage of white pixels in the image
    """     

    if image.shape.size!=3 or image.shape[2]!=3:
        raise ValueError("image has to be an array of shape (None, None, 3)")
            
    height = image.shape[0]
    width = image.shape[1]
    total = height * width
    white=0
    for h in range(height):
        for w in range(width):
            pix = image[h][w]
            if np.array_equal(pix, [255, 255, 255]):
                white +=1
    return white/total

train_images = os.path.join(TRAIN_FOLDER, "sat")
removed_train_images = os.path.join(TRAIN_FOLDER, "removed_sat")

train_maps = os.path.join(TRAIN_FOLDER, "map")
removed_train_maps = os.path.join(TRAIN_FOLDER, "removed_map")

for filename in os.listdir(train_images):
    full_filename = os.path.join(train_images, filename)
    im = Image.open(full_filename)
    im = np.array(im)
    perc= whitePixelsPercentage(im)
    if perc > 0.3:
        print("Removing " + full_filename)
        dest_img = os.path.join(removed_train_images, filename)
        
        src_map = os.path.join(train_maps, filename[:-1])
        dest_map = os.path.join(removed_train_maps, filename[:-1])
        os.rename(full_filename, dest_img)
        os.rename(src_map, dest_map)