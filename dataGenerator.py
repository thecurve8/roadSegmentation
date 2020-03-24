# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:34:56 2020

@author: Alexander

This script handles the data generator for the model bases on the massashussets dataset

PIL and numpy are required
"""

import os
from PIL import Image
import numpy as np
from settings import MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH

def generate(foldername, BatchSize, threshold=200):
    """Generator for batches of 400x400 images and labels from a specified folder
    
    Takes images and corresponding labels and splits them in 400x400 squares.
    If total width or height are not a multiple of 400, last column or row covers part of previous one
    
    Parameters
    ----------
    foldername : string
        folder containing "sat" and "map" folders containg images and labels
    BatchSize : int
        size of each batch
    threshold : int, optional
        threshold used for labels to set a pixel as a road (default is 200)
        
    Returns
    -------
    (images, labels): (np.array (float32), np.array (float32))
        numpy arrays ((BatchSize, 400, 400, 3), (BatchSize, 400, 400))
        RGB[0...255, 0...255, 0...255], BW[0...1]
    """     
    
    images_filename = os.path.join(foldername, "sat")
    labels_filename = os.path.join(foldername, "map")
    while True:
        images=[]
        labels=[]
        for filename in os.listdir(images_filename):
            filepath_image = os.path.join(images_filename, filename)
            im = Image.open(filepath_image)
            im = np.array(im)
            
            filepath_groundtruth = os.path.join(labels_filename, filename[:-1])
            groundtruth = Image.open(filepath_groundtruth)
            groundtruth = np.array(groundtruth)
            groundtruth = np.where(groundtruth > threshold, 1, 0)
            
            height = im.shape[0]
            width = im.shape[1]
            if height<MODEL_INPUT_HEIGHT or width<MODEL_INPUT_WIDTH:
                raise ValueError("images have to be at least 400x400")
                
            for i in range(0, height, MODEL_INPUT_HEIGHT):
                if i+MODEL_INPUT_HEIGHT>height:
                    i=height-MODEL_INPUT_HEIGHT
                for j in range(0,width, MODEL_INPUT_WIDTH):
                    if j+MODEL_INPUT_WIDTH>width:
                        j=width-MODEL_INPUT_WIDTH
                    
            
                    if len(images)>=BatchSize:
                        images=[]
                        labels=[]
                    im_patch = im[i:i+MODEL_INPUT_HEIGHT, j:j+MODEL_INPUT_WIDTH]
                    lab_patch = groundtruth[i:i+MODEL_INPUT_HEIGHT, j:j+MODEL_INPUT_WIDTH]
                    images.append(im_patch)
                    labels.append(lab_patch)
                    
                    if len(images)==BatchSize:
                        yield (np.array(images).astype(np.float32), np.array(labels).astype(np.float32))