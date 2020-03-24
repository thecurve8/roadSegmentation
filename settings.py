# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:02:38 2020

@author: Alexander

This script is used for global settings
"""
import os

"""
MODEL_INPUT_HEIGHT and MODEL_INPUT_WIDTH should be multiples of 5
to make sure input and output have same size in the model
"""

MODEL_INPUT_HEIGHT=400
MODEL_INPUT_WIDTH=400
UP_FOLDER = "mass_roads"
TRAIN_FOLDER =os.path.join(UP_FOLDER, "train")
VALIDATION_FOLDER = os.path.join(UP_FOLDER, "valid")
TEST_FOLDER = os.path.join(UP_FOLDER, "test")
MODELS_FOLDER = "models"
TRAIN_SIZE = 885
VALIDATION_SIZE = 14
TEST_SIZE = 49