# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:39:18 2020

@author: Alexander
"""
from dataGenerator import generate
import os
import numpy as np
from settings import TRAIN_FOLDER
# up_folder = "mass_roads"
# train_folder =os.path.join(up_folder, "valid")
# TRAIN_SIZE=200

# BS_TRAIN = 20
# TRAIN_STEPS_PER_EPOCH = TRAIN_SIZE/BS_TRAIN

# train_gen = generate(train_folder, BS_TRAIN)
# n= next(train_gen)
# (x,y) = n
# x_6 = x[6] 
# for b in range(20):
#     for i in range(400):
#         for j in range(400):
#             if y[b,i,j]>0 and y[b,i,j]<255:
#                 print(y[b,i,j])
# labels_filename = os.path.join(TRAIN_FOLDER, "map")
# files = os.listdir(labels_filename)
# files = np.array(files)
# randIndx = np.arange(len(files))
# np.random.shuffle(randIndx)
# # print(randIndx)
# shuffledFiles = files[randIndx]
# for filename in shuffledFiles:
#     print(filename)
