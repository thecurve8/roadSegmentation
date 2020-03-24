# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:02:20 2020

@author: Alexander

[UNUSED] 
This script is used to load all the data from the EPFL dataset
"""

# import os
# import numpy as np
# from settings import PATH_TRAINING_GROUNDTRUTH, PATH_TRAINING_IMAGES, TRAINING_SIZE, TEST_SIZE
# from PIL import Image

# def loadTrainData():
#     images = []
#     groundtruths=[]
#     for i in range(1, TRAINING_SIZE+1):
#         try:
#             filepath_image = os.path.join(PATH_TRAINING_IMAGES, 'satImage_' + '%.3d' % i + '.png')
#             filepath_groundtruth = os.path.join(PATH_TRAINING_GROUNDTRUTH, 'satImage_' + '%.3d' % i + '.png')
#             image = Image.open(filepath_image)
#             image = np.array(image)
            
#             threshold=200
#             groundtruth = Image.open(filepath_groundtruth)
#             groundtruth = np.array(groundtruth)
#             groundtruth = np.where(groundtruth > threshold, 1, 0)            
            
#             images.append(image)
#             groundtruths.append(groundtruth)
#         except IOError: 
#             print("Error while loading ", filepath_image)
#     images=np.array(images)
#     groundtruths=np.array(groundtruths)
#     groundtruths = np.expand_dims(groundtruths, -1)
#     return images, groundtruths

# def loadTestData():
#     images = []
#     for i in range(1, TEST_SIZE+1):
#         try:
#             filepath_image = os.path.join("test_set_images", "test_"+str(i))
#             filepath_image = os.path.join(filepath_image, 'test_' + str(i) + '.png')
#             image = Image.open(filepath_image)
#             image = image.resize((400,400))
#             image = np.array(image)
            
#             images.append(image)
#         except IOError: 
#             print("Error while loading ", filepath_image)
#     images=np.array(images)
#     return images


