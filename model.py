# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:58:21 2020

@author: Alexander

This script is used to create the model for road segmentation, to train it and
to see the results.

Keras, PIL, numpy and Matplotlib are used.
"""

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPool2D, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from settings import MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODELS_FOLDER, TRAIN_FOLDER, VALIDATION_FOLDER, TEST_FOLDER
from dataGenerator import generate

def dice_coef(y_true, y_pred, smooth = 1):
    """Dice coeficient = (2* |X inter Y| + smooth)/(|X|+|Y|+smooth)
    
    Parameters
    ----------
    y_true : np.array (float32)
        batch of labels
    y_pred : np.array (flot 32)
        batch of predictions
    smooth : int, optional
        smoothing value to avoir devisions by 0 and to make sure (default is 1)
        
    Returns
    -------
    dice_coef: float
        value is in [0,1]
    """     
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    """Dice loss = 1-(2* |X inter Y| + smooth)/(|X|+|Y|+smooth)
    
    Parameters
    ----------
    y_true : np.array (float32)
        batch of labels
    y_pred : np.array (flot 32)
        batch of predictions
        
    Returns
    -------
    dice_coef: float
        value is in [0,1]
    """     

    return 1-dice_coef(y_true, y_pred)

def get_deep_model():
    """Creates a vanilla U-Network
    
    Input is (None, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 3), float32, RGB[0...255]
    Output is (None, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 1), float32, BW[0...1]

    U-Net has:
    -5 step encoder 
        2 conv3x3 with Relu and normalization at each step
        max_pool 2X2 between each step
    -5 step decoder
        2 conv3x3 with Relu and normalization at each step
        up_conv 2x2 between each step
        
    Returns
    -------
    model: keras.Model
        U-Net
    """     

    inputs = Input(shape=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 3))
    
    ####ENCODER###########
    
    conv1 = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(16, (3,3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    max_pool1 = MaxPool2D((2,2))(conv1)
    
    conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(max_pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    max_pool2 = MaxPool2D((2,2))(conv2)
    
    conv3 = Conv2D(64, (3,3), activation='relu', padding='same')(max_pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(64, (3,3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    max_pool3 = MaxPool2D((2,2))(conv3)
    
    conv4 = Conv2D(128, (3,3), activation='relu', padding='same')(max_pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(128, (3,3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    max_pool4 = MaxPool2D((2,2))(conv4)
    
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same')(max_pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    ###DECODER############
    upsample6 = Conv2DTranspose(128, (2,2), strides=(2,2), activation='relu', padding='same')(conv5)
    conv6 = concatenate([upsample6, conv4])
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    upsample7 = Conv2DTranspose(128, (2,2), strides=(2,2), activation='relu', padding='same')(conv6)
    conv7 = concatenate([upsample7, conv3])
    conv7 = Conv2D(64, (3,3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(64, (3,3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    upsample8 = Conv2DTranspose(128, (2,2), strides=(2,2), activation='relu', padding='same')(conv7)
    conv8 = concatenate([upsample8, conv2])
    conv8 = Conv2D(32, (3,3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(32, (3,3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    upsample9 = Conv2DTranspose(128, (2,2), strides=(2,2), activation='relu', padding='same')(conv8)
    conv9 = concatenate([upsample9, conv1])
    conv9 = Conv2D(16, (3,3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(16, (3,3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid', padding='same')(conv9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def latestModel(model_name="trainedModel_deep"):
    """Finds the latest saved model for the U-Net
    
    Models are saved in the models folder. 
    If such a folder doesn't exist it will be created by this function.
    File should be of the format NNN_XXX.h5 
    Where NNN is the model name.
    Where XXX is the epoch ehen it was saved.
    Params
    ------
    model_name: string, optional
        name of the file to save this model (default "trainedModel_deep")
        
    Returns
    -------
    found, file_path, epoch: bool, string, int
        found is True if the file exists,
        file_path is the path to the file from the local directory,
        epoch is the epoch when the model was saved
    """     

    if not(os.path.isdir(MODELS_FOLDER)):
        print("Creating ./models to store saved models")
        os.mkdir(MODELS_FOLDER)
        return False, "", 0
    
    onlyfiles = [f for f in os.listdir(MODELS_FOLDER) if os.path.isfile(os.path.join(MODELS_FOLDER, f))]
    biggest_step=-1
    file_with_biggest_step=""
    for file in onlyfiles:
        filename, file_extension = os.path.splitext(file)
        beginning = model_name+"_"
        if file_extension==".h5" and filename.startswith(beginning):
            rest=filename[len(beginning):]
            try:
                int_value = int(rest)
                if int_value > biggest_step:
                    biggest_step=int_value
                    file_with_biggest_step=filename+file_extension
            except ValueError:
                pass
    if biggest_step!=-1:
        print("Biggest step found is ", biggest_step)
        print("Model file is " + file_with_biggest_step)
        file_path = os.path.join(MODELS_FOLDER, file_with_biggest_step)
        return True, file_path, biggest_step
    else:
        return False, "", 0
        
def train_deep_UNet(epochs_to_run=30, model_name="trainedModel_deep"):
    """Trains the latest model or starts a new traiing session if nothing found
    
    Models are saved in the models folder. 
    If such a folder doesn't exist it will be created by this function.
    File should be of the format NNN_XXX.h5 
    Where NNN is the model name.
    Where XXX is the epoch ehen it was saved.
    
    Params
    ------
    epochs_to_run: int, optional
        epochs to continue traing for (default 30)
    model_name: string, optional
        name of the model to train (default "trainedModel_deep")
    """     

    found, path, trainedEpochs = latestModel(model_name)
    if not found:
        model = get_deep_model()
    else:
        model = load_model(path, compile=False)
    # model.summary()
    model.compile(optimizer="adam", loss=soft_dice_loss)

    TRAIN_SIZE=200
    VALIDATION_SIZE=14
    BS_TRAIN = 20
    TRAIN_STEPS_PER_EPOCH = TRAIN_SIZE/BS_TRAIN
    
    train_gen = generate(TRAIN_FOLDER, BS_TRAIN)
    val_gen = generate(VALIDATION_FOLDER, VALIDATION_SIZE)
    history = model.fit(train_gen,
                    steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
                    validation_data = val_gen,
                    validation_steps = 1,
                    epochs=epochs_to_run)
    saved_model_name = model_name+"_"+(trainedEpochs+epochs_to_run)+".h5"
    model.save(saved_model_name)


def test_model(model_name="trainedModel_deep"):
    """Tests the latest model 
    
    Models are saved in the models folder. 
    If such a folder doesn't exist it will be created by this function.
    File should be of the format NNN_XXX.h5 
    Where NNN is the model name.
    Where XXX is the epoch ehen it was saved.
    
    Params
    ------
    model_name: string, optional
        name of the model to train (default "trainedModel_deep")
        
    Returns
    -------
    predictions: np.array float32
        np.array with predictions of shape (None, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)
        encodes for BW [0...1]
    """     

    images_filename = os.path.join(TEST_FOLDER, "sat")
    labels_filename = os.path.join(TEST_FOLDER, "map")
    images=[]
    labels=[]
    for filename in os.listdir(images_filename):
        filepath_image = os.path.join(images_filename, filename)
        im = Image.open(filepath_image)
        im = np.array(im)
        
        filepath_groundtruth = os.path.join(labels_filename, filename[:-1])
        groundtruth = Image.open(filepath_groundtruth)
        groundtruth = np.array(groundtruth)
        groundtruth = np.where(groundtruth > 200, 1, 0)
        
        height = im.shape[0]
        width = im.shape[1]
        for i in range(0, height, MODEL_INPUT_HEIGHT):
                if i+MODEL_INPUT_HEIGHT>height:
                    i=height-MODEL_INPUT_HEIGHT
                for j in range(0,width, MODEL_INPUT_WIDTH):
                    if j+MODEL_INPUT_WIDTH>width:
                        j=width-MODEL_INPUT_WIDTH
                    im_patch = im[i:i+MODEL_INPUT_HEIGHT, j:j+MODEL_INPUT_WIDTH]
                    lab_patch = groundtruth[i:i+MODEL_INPUT_HEIGHT, j:j+MODEL_INPUT_WIDTH]
                    images.append(im_patch)
                    labels.append(lab_patch)
                
    images, labels =np.array(images), np.array(labels)
    
    found, path, trainedEpochs = latestModel(model_name)
    if not found:
        raise ValueError(model_name +" is not found in the ./models directory")
    else:
        model = load_model(path, compile=False)
    model.compile(optimizer='adam', loss=soft_dice_loss)

    image_to_feed=images[48:48+20]
    labels_to_feed = labels[48:48+20]
    prediction = model.predict([image_to_feed])
    plt.figure(figsize=(40,32))
    
    x, y = 5,4
    for i in range(y):  
        for j in range(x):
            plt.subplot(y*3, x, i*3*x+j+1)
            pos = i*5 + j
            plt.imshow(image_to_feed[pos])
            plt.title('Sat img #{}'.format(pos))
            plt.axis('off')
            
            plt.subplot(y*3, x, (i*3+1)*x+j+1)
            # We display the associated mask we just generated above with the training image
            plt.imshow(np.squeeze(prediction[pos]))
            plt.title('Mask #{}'.format(pos))
            plt.axis('off')
            
            plt.subplot(y*3, x, (i*3+2)*x+j+1)
            # We display the associated groundtruth
            plt.imshow(labels_to_feed[pos])
            plt.title('Groundtruth #{}'.format(pos))
            plt.axis('off')
            
    plt.show()
    return np.squeeze(prediction, axis=-1)
    
if __name__ =="__main__":
    test_model()