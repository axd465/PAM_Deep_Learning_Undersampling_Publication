# Import Statements:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import scipy.io as sio
import h5py
import random
import shutil
import PIL
import imageio
import keras.backend as K
from pathlib import Path
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Add, Input, Dropout
from tensorflow.keras.layers import BatchNormalization, UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.layers import Activation

###################################################################################################
'''
MODEL DEFINITION:
Modified UNet
'''
# Original UNet based on https://www.tandfonline.com/doi/full/10.1080/17415977.2018.1518444?af=R
def ICL_ResUNet(input, filters = 32, kernel_size = 3, padding = 'same', 
                activation = 'relu', kernel_initializer = 'glorot_normal'):
    shortcut1_1 = input
    [out, shortcut1_2] = DownBlock(input, filters, kernel_size, padding, activation, kernel_initializer)
    [out, shortcut2_1] = DownBlock(out, filters*2, kernel_size, padding, activation, kernel_initializer)
    [out, shortcut3_1] = DownBlock(out, filters*2*2, kernel_size, padding, activation, kernel_initializer)
    [out, shortcut4_1] = DownBlock(out, filters*2*2*2, kernel_size, padding, activation, kernel_initializer)
    
    
    out = UpBlock(out, filters*2*2*2*2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut4_1])
    out = UpBlock(out, filters*2*2*2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut3_1])
    out = UpBlock(out, filters*2*2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut2_1])
    out = UpBlock(out, filters*2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut1_2])
    
    '''
    out = ICL_Res_Block(input, filters, kernel_size, strides=1, padding=padding, 
                        activation=activation, kernel_initializer = kernel_initializer, prob=0.05)
    '''
    #'''
    out = Conv2D_BatchNorm(out, filters, kernel_size, strides = 1, padding=padding, 
                           activation=activation, kernel_initializer = kernel_initializer)
    out = Conv2D_BatchNorm(out, filters, kernel_size, strides = 1, padding=padding, 
                           activation=activation, kernel_initializer = kernel_initializer)
    #'''
    # 1x1 Convolution Followed by Identity as Activation:
    '''
    # WITH ICL
    out = Conv2D_with_ICL(out, filters=1, kernel_size=1, strides=1, padding=padding, 
                          activation='linear', kernel_initializer=kernel_initializer, prob=0.05)
    '''
    '''
    # WITH BATCHNORM
    out = Conv2D_BatchNorm(out, filters=1, kernel_size=1, strides=1, padding=padding, 
                           activation='linear', kernel_initializer=kernel_initializer)
    '''
    #'''
    # Without BatchNorm
    out = Conv2D(filters=1, kernel_size=1, 
                 strides=1, padding=padding, 
                 activation='linear', 
                 kernel_initializer=kernel_initializer)(out)
    #'''
    out = Add()([out, shortcut1_1])
    return out

def DownBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    #print('DOWN_in: '+str(input.shape))
    ############################################
    # Modified UNet - ICL RES Down Block:
    out = ICL_Res_Block(input, filters, kernel_size, strides=1, padding=padding, 
                        activation=activation, kernel_initializer = kernel_initializer, prob=0.05)
    shortcut = out
    out = DownSample(out, filters, kernel_size, strides=2, padding=padding, 
                     activation=activation, kernel_initializer = kernel_initializer)
    ############################################
    #print('DOWN_out: '+str(out.shape))
    return [out, shortcut]

def UpBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    #print('UP_in: '+str(input.shape))
    #print(filters)
    ############################################
    # Modified UNet - ICL RES Up Block:
    out = Conv2D_BatchNorm(input, filters, kernel_size, strides = 1, padding=padding, 
                           activation=activation, kernel_initializer = kernel_initializer)
    out = Conv2D_BatchNorm(out, filters, kernel_size, strides = 1, padding=padding, 
                           activation=activation, kernel_initializer = kernel_initializer)
    #'''
    out = UpSample(out, filters//2, kernel_size, strides=2,padding=padding, 
                   activation=activation, kernel_initializer=kernel_initializer)
    #'''
    '''
    out = Conv2D_Transpose_BatchNorm(out, filters//2, kernel_size, strides=2, padding=padding, 
                                     activation=activation, kernel_initializer=kernel_initializer)
    '''
    '''
    out = Conv2D_Transpose_ICL(out, filters//2, kernel_size, strides=2, padding=padding, 
                               activation=activation, kernel_initializer=kernel_initializer, prob=0.05)
    '''
    ############################################
    #print('UP_out: '+str(out.shape))
    return out
###################################################################################################
'''
MODEL FUNCTIONS:
'''
def Conv2D_BatchNorm(input, filters, kernel_size=3, strides=1, padding='same',
                     activation='linear', kernel_initializer='glorot_normal'):
    out = Conv2D(filters=filters, kernel_size=kernel_size, 
                 strides=strides, padding=padding, 
                 activation=activation, 
                 kernel_initializer=kernel_initializer)(input)
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, 
                             scale=True, beta_initializer='zeros', gamma_initializer='ones', 
                             moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, 
                             gamma_constraint=None)(out)
    return out


def Upsample_Conv2D_ICL(input, filters, kernel_size=3, strides=1, padding='same', 
                        activation='linear', kernel_initializer='glorot_normal', prob=0.05):
    out = UpSampling2D(size=2)(input) # Upsampling ratio of two
    out = Conv2D_with_ICL(out, filters, kernel_size=kernel_size, strides=strides, padding=padding,
                          activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    return out

def Conv2D_Transpose_ICL(input, filters, kernel_size=3, strides=2, padding='same', 
                         activation='relu', kernel_initializer='glorot_normal', prob=0.05):
    out = Independent_Component_Layer(input, prob=prob)
    # Conv2DTranspose also known as a 2D Deconvolution
    out = Conv2DTranspose(filters, kernel_size, strides=2, padding=padding,
                          activation=activation, kernel_initializer=kernel_initializer)(out)
    return out

def Conv2D_Transpose_BatchNorm(input, filters, kernel_size=3, strides=2, padding='same', 
                               activation='relu', kernel_initializer='glorot_normal'):
    # Conv2DTranspose also known as a 2D Deconvolution
    out = Conv2DTranspose(filters, kernel_size, strides=2, padding=padding, 
                          activation=activation, kernel_initializer=kernel_initializer)(input)
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, 
                             scale=True, beta_initializer='zeros', gamma_initializer='ones', 
                             moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, 
                             gamma_constraint=None)(out)
    return out
def DownSample(input, filters, kernel_size=3, strides=2, padding='same',
               activation='linear', kernel_initializer='glorot_normal'):
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, padding=padding, 
                           activation=activation, kernel_initializer = kernel_initializer)
    out = Conv2D_BatchNorm(out, filters, kernel_size, strides=strides, padding=padding, 
                           activation=activation, kernel_initializer = kernel_initializer)
    return out
def UpSample(input, filters, kernel_size=3, strides=2, padding='same',
             activation='linear', kernel_initializer='glorot_normal'):
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, padding=padding, 
                           activation=activation, kernel_initializer = kernel_initializer)
    out = Conv2D_Transpose_BatchNorm(out, filters//2, kernel_size, strides=strides, padding=padding, 
                                     activation=activation, kernel_initializer=kernel_initializer)
    return out

###################################################################################################
# NOT USED:
def Upsample_Conv2D_BatchNorm(input, filters, kernel_size=3, strides=1, padding='same', 
                              activation='linear', kernel_initializer='glorot_normal'):
    out = UpSampling2D(size=2)(input) # Upsampling ratio of two
    out = Conv2D_BatchNorm(out, filters, kernel_size=kernel_size, strides=strides, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)
    return out

###################################################################################################
'''
INDEPENDENT COMPONENT LAYER:
'''
# Inpired by work done in https://arxiv.org/abs/1905.05928v1:
def Independent_Component_Layer(input, prob=0.05):
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, 
                             scale=True, beta_initializer='zeros', gamma_initializer='ones', 
                             moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, 
                             gamma_constraint=None)(input)
    out = Dropout(prob)(out)
    return out
def Conv2D_with_ICL(input, filters, kernel_size=3, strides=1, padding='same',
                    activation='linear', kernel_initializer='glorot_normal', prob=0.05):
    out = Independent_Component_Layer(input, prob)
    out = Conv2D(filters=filters, kernel_size=kernel_size, 
                 strides=strides, padding=padding, 
                 activation=activation, 
                 kernel_initializer=kernel_initializer)(out)
    return out
def ICL_Res_Block(input, filters, kernel_size=3, strides=1, padding='same', 
                  activation='relu', kernel_initializer='glorot_normal', prob=0.05):
    # Residual Connection
    out = Activation(activation)(input)
    residual = Conv2D_with_ICL(out, filters, kernel_size=1, strides=1, padding=padding, 
                               activation='linear', kernel_initializer = kernel_initializer, prob=0.05)
    # Res Block
    out = Activation(activation)(input)
    out = Conv2D_with_ICL(out, filters, kernel_size, strides=1, padding=padding, 
                          activation='linear', kernel_initializer = kernel_initializer, prob=0.05)
    out = Activation(activation)(out)
    out = Conv2D_with_ICL(out, filters, kernel_size, strides=1, padding=padding, 
                          activation='linear', kernel_initializer = kernel_initializer, prob=0.05)
    out = Add()([residual, out])
    return out
###################################################################################################
'''
FUNCTION TO INSTANTIATE MODEL:
'''
def getModel(input_shape, filters, kernel_size, padding='same', activation='relu', kernel_initializer='glorot_normal'):
    model_inputs = Input(shape=input_shape, name='img')
    model_outputs = ICL_ResUNet(model_inputs, filters=filters, kernel_size=kernel_size, padding=padding, 
                                activation=activation, kernel_initializer=kernel_initializer)
    model = Model(model_inputs, model_outputs, name='ICL_ResUNet_Model')
    return model