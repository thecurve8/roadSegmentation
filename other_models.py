# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:21:45 2020

@author: Alexander
"""
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPool2D, BatchNormalization, Conv2DTranspose, concatenate
from settings import MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH
from dataLoader import loadTrainData, loadTestData
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def train_model():
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
    model.summary()
    
    train_images, train_labels = loadTrainData()
    
    
    plt.figure(figsize=(20,16))
    
    x, y = 5,4
    for i in range(y):  
        for j in range(x):
            plt.subplot(y*2, x, i*2*x+j+1)
            pos = i*4 + j
            plt.imshow(train_images[pos])
            plt.title('Sat img #{}'.format(pos))
            plt.axis('off')
            plt.subplot(y*2, x, (i*2+1)*x+j+1)
               
            #We display the associated mask we just generated above with the training image
            plt.imshow(np.squeeze(train_labels[pos]))
            plt.title('Mask #{}'.format(pos))
            plt.axis('off')
            
    plt.show()
    model.compile(optimizer="adam", loss=soft_dice_loss)
    history = model.fit(train_images,
                    train_labels,
                    validation_split = 0.1,
                    epochs=30,
                    batch_size = 10
                       )
    model.save("trainedModel_U.h5")



def train_small():
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
    
    ###DECODER############
    
    upsample3 = Conv2DTranspose(16, (2,2), strides=(2,2), activation='relu', padding='same')(conv2)
    conv3 = concatenate([upsample3, conv1])
    conv3 = Conv2D(16, (3,3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(16, (3,3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid', padding='same')(conv3)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    
    
    
    
    train_images, train_labels = loadTrainData()
    
    
    plt.figure(figsize=(20,16))
    
    x, y = 5,4
    for i in range(y):  
        for j in range(x):
            plt.subplot(y*2, x, i*2*x+j+1)
            pos = i*4 + j
            plt.imshow(train_images[pos])
            plt.title('Sat img #{}'.format(pos))
            plt.axis('off')
            plt.subplot(y*2, x, (i*2+1)*x+j+1)
               
            #We display the associated mask we just generated above with the training image
            plt.imshow(np.squeeze(train_labels[pos]))
            plt.title('Mask #{}'.format(pos))
            plt.axis('off')
            
    plt.show()
    model.compile(optimizer="adam", loss=soft_dice_loss)
    history = model.fit(train_images,
                    train_labels,
                    validation_split = 0.1,
                    epochs=30,
                    batch_size = 10
                       )
    model.save("trainedModel_1.h5")
    
def test_model():
    model = load_model("trainedModel_deep.h5", compile=False)
    model.compile(optimizer='adam', loss=soft_dice_loss)
    images = loadTestData()
    image_to_feed=images[0:5]#np.expand_dims(images[0], axis=-1)
    prediction = model.predict([image_to_feed])
    for i in range(5):
        img = Image.fromarray(images[i], 'RGB')
        img.show()
        pre = np.squeeze(prediction[i],-1)
        pre = binary_to_uint8(pre)
        pre = Image.fromarray(pre)
        pre.show()
        
def test_model_1():
    model = load_model("trainedModel_deep_60.h5", compile=False)
    model.compile(optimizer='adam', loss=soft_dice_loss)
    images = loadTestData()
    image_to_feed=images[0:20]#np.expand_dims(images[0], axis=-1)
    prediction = model.predict([image_to_feed])
    plt.figure(figsize=(20,16))
    
    x, y = 5,4
    for i in range(y):  
        for j in range(x):
            plt.subplot(y*2, x, i*2*x+j+1)
            pos = i*5 + j
            plt.imshow(images[pos])
            plt.title('Sat img #{}'.format(pos))
            plt.axis('off')
            plt.subplot(y*2, x, (i*2+1)*x+j+1)
               
            #We display the associated mask we just generated above with the training image
            plt.imshow(np.squeeze(prediction[pos]))
            plt.title('Mask #{}'.format(pos))
            plt.axis('off')
            
    plt.show()
    return prediction