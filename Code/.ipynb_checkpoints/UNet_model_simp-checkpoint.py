# customary imports:
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
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
from functools import reduce
# Based on https://www.tandfonline.com/doi/full/10.1080/17415977.2018.1518444?af=R
class UNet_model_simp(tf.keras.Model):
    def __init__(self, filters = 32, kernel_size = 3, activation = 'relu'):
        super(UNet_model_simp, self).__init__(name = 'UNet_model_simp')
        self.UNet_DownBlock1 = UNet_DownBlock(filters, kernel_size, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        self.UNet_DownBlock2 = UNet_DownBlock(filters*2, kernel_size, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        self.UNet_DownBlock3 = UNet_DownBlock(filters*2*2, kernel_size, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        self.UNet_DownBlock4 = UNet_DownBlock(filters*2*2*2, kernel_size, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        
        self.UNet_UpBlock1 = UNet_UpBlock(filters*2*2*2*2, kernel_size, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        self.UNet_UpBlock2 = UNet_UpBlock(filters*2*2*2, kernel_size, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        self.UNet_UpBlock3 = UNet_UpBlock(filters*2*2, kernel_size, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        self.UNet_UpBlock4 = UNet_UpBlock(filters*2, kernel_size, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        
        # Format: convD_N -> where D is the U-Net architecture depth and N is the # of conv at that depth
        self.conv1_3 = Conv2D_BatchNorm(filters, kernel_size, strides=1, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        self.conv1_4 = Conv2D_BatchNorm(filters, kernel_size, strides=1, padding='same', activation=activation, kernel_initializer = 'glorot_normal')
        self.conv1_5 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 1, strides=1, padding='same', kernel_initializer = 'glorot_normal')
        
        self.residual = tf.keras.layers.Concatenate()
        self.summation = tf.keras.layers.Add()
        
        # Defining the Loss Function
        self.loss_func = self.model_loss()
    
    def call(self, input, training=None):
        #shortcut1_1 = input
        #print('input shape = '+str(input.shape))
        shortcut1_1 = input
        shortcut1_2, out = self.UNet_DownBlock1(shortcut1_1)
        
        shortcut2_1, out = self.UNet_DownBlock2(out)
        
        shortcut3_1, out = self.UNet_DownBlock3(out)
        
        shortcut4_1, out = self.UNet_DownBlock4(out)
        
        out = self.UNet_UpBlock1(out)
        
        out = self.residual([out, shortcut4_1])
        out = self.UNet_UpBlock2(out)
        
        out = self.residual([out, shortcut3_1])
        out = self.UNet_UpBlock3(out)
        
        out = self.residual([out, shortcut2_1])
        out = self.UNet_UpBlock4(out)
        
        out = self.residual([out, shortcut1_2])
        out = self.conv1_3(out)
        out = self.conv1_4(out)
        out = self.conv1_5(out)
        out = self.summation([out, shortcut1_1])
        #print('out shape = '+str(out.shape))
        return out
    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """
        recon_loss_func = tf.keras.losses.mean_absolute_error
        def total_loss(y_true, y_pred):
            """ Final loss calculation function to be passed to optimizer"""
            # Reconstruction loss
            recon_loss = recon_loss_func(y_true, y_pred)
            loss = recon_loss
            return loss
        total_loss.__name__ = "total_loss"
        return total_loss
    def get_config(self):
        base_config = super(UNet_model_simp, self).get_config()
        base_config['loss_func'] = self.loss_func
        base_config['UNet_DownBlock1'] = self.UNet_DownBlock1
        base_config['UNet_DownBlock2'] = self.UNet_DownBlock2
        base_config['UNet_DownBlock3'] = self.UNet_DownBlock3
        base_config['UNet_DownBlock4'] = self.UNet_DownBlock4
        base_config['UNet_UpBlock1'] = self.UNet_UpBlock1
        base_config['UNet_UpBlock2'] = self.UNet_UpBlock2
        base_config['UNet_UpBlock3'] = self.UNet_UpBlock3
        base_config['UNet_UpBlock4'] = self.UNet_UpBlock4
        base_config['conv1_3'] = self.conv1_3
        base_config['conv1_4'] = self.conv1_4
        base_config['conv1_5'] = self.conv1_5
        base_config['residual'] = self.residual
        base_config['summation'] = self.summation
        return base_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
       
class UNet_DownBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = 3, padding = 'same', activation = 'relu', kernel_initializer = 'glorot_normal'):
        super(UNet_DownBlock, self).__init__(name = 'UNet_DownBlock')
        self.conv1 = Conv2D_BatchNorm(filters, kernel_size, strides=1, padding=padding, activation=activation, kernel_initializer=kernel_initializer)
        self.conv2 = Conv2D_BatchNorm(filters, kernel_size, strides=1, padding=padding, activation=activation, kernel_initializer=kernel_initializer)
        self.max_pool = MaxPooling2D(pool_size = (2,2)) # Need Max Pool Instead of Stride = 2 for Shortcut Connection
    def call(self, input):
        #print('input')
        #print(input.shape)
        out = self.conv1(input)
        out = self.conv2(out)
        shortcut = out
        out = self.max_pool(out)
        #print('output')
        #print(out.shape)
        return [shortcut, out]
    def get_config(self):
        base_config = super(UNet_DownBlock, self).get_config()
        base_config['conv1'] = self.conv1
        base_config['conv2'] = self.conv2
        base_config['max_pool'] = self.max_pool
        return base_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class UNet_UpBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = 3, padding = 'same', activation = 'relu', kernel_initializer = 'glorot_normal'):
        super(UNet_UpBlock, self).__init__(name = 'UNet_UpBlock')
        self.conv1 = Conv2D_BatchNorm(filters, kernel_size, strides=1, padding=padding, activation=activation, kernel_initializer=kernel_initializer)
        self.conv2 = Conv2D_BatchNorm(filters, kernel_size, strides=1, padding=padding, activation=activation, kernel_initializer=kernel_initializer)
        #self.conv3 = Conv2D_Transpose_BatchNorm(filters, kernel_size, strides=2, padding=padding, activation=activation, kernel_initializer=kernel_initializer)
        self.conv3 = Upsample_Conv2D_BatchNorm(filters//2, kernel_size, strides=1, upsample_ratio = 2, padding=padding,
                                               activation=activation, kernel_initializer=kernel_initializer)
    def call(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
    def get_config(self):
        base_config = super(UNet_UpBlock, self).get_config()
        base_config['conv1'] = self.conv1
        base_config['conv2'] = self.conv2
        base_config['conv3'] = self.conv3
        return base_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class Conv2D_BatchNorm(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = 'glorot_normal'):
        super(Conv2D_BatchNorm, self).__init__(name = 'Conv2D_BatchNorm')
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                             activation=activation, kernel_initializer=kernel_initializer)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, center=True, scale=True, 
                                                             beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
                                                             beta_constraint=None, gamma_constraint=None)
    def call(self, input, training = None):
        if training is None:
            train = True
            #print(train)
        elif training is False:
            train = False
            #print(train)
        else:
            train = True
            #print(train)
        out = self.conv2d(input)
        out = self.batch_norm(out, training = train)
        return out
    def get_config(self):
        base_config = super(Conv2D_BatchNorm, self).get_config()
        base_config['conv2d'] = self.conv2d
        base_config['batch_norm'] = self.batch_norm
        return base_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class Upsample_Conv2D_BatchNorm(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = 3, strides = 1, upsample_ratio = 2, padding = 'same', activation = 'relu', kernel_initializer = 'glorot_normal'):
        super(Upsample_Conv2D_BatchNorm, self).__init__(name = 'Upsample_Conv2D_BatchNorm')
        self.upsample2d = tf.keras.layers.UpSampling2D(size=upsample_ratio)
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                             activation=activation, kernel_initializer=kernel_initializer)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, center=True, scale=True, 
                                                             beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
                                                             beta_constraint=None, gamma_constraint=None)
    def call(self, input, training = None):
        if training is None:
            train = True
            #print(train)
        elif training is False:
            train = False
            #print(train)
        else:
            train = True
            #print(train)
        out = self.upsample2d(input)
        out = self.conv2d(out)
        out = self.batch_norm(out, training = train)
        return out
    def get_config(self):
        base_config = super(Upsample_Conv2D_BatchNorm, self).get_config()
        base_config['upsample2d'] = self.upsample2d
        base_config['conv2d'] = self.conv2d
        base_config['batch_norm'] = self.batch_norm
        return base_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# class Conv2D_Transpose_BatchNorm(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = 'glorot_normal'):
#         super(Conv2D_Transpose_BatchNorm, self).__init__(name = 'Conv2D_Transpose_BatchNorm')
#         self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
#                                                                 activation=activation, kernel_initializer=kernel_initializer)
#         self.batch_norm = None
#     def call(self, input, training):
#        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
#                                                             beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
#                                                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
#                                                             beta_constraint=None, gamma_constraint=None, trainable = training)
#         out = self.conv2d_transpose(input)
#         out = self.batch_norm(out)
#         return out
#     def get_config(self):
#         base_config = super(Conv2D_Transpose_BatchNorm, self).get_config()
#         base_config['conv2d_transpose'] = self.conv2d_transpose
#         base_config['batch_norm'] = self.batch_norm
#         return base_config
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    