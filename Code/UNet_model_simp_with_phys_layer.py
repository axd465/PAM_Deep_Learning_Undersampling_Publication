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
class UNet_model_simp_with_phys_layer(tf.keras.Model):
    def __init__(self, filters = 32, kernel_size = 3, activation = 'relu', init_alpha = 1, growth_rate = 1.001, 
                 alpha_schedule = 5, gamma = 0.1, sparsity = 0.75, with_phys_layer = True, with_learn_phys = True,
                 downsample_axis = 'x'):
        super(UNet_model_simp_with_phys_layer, self).__init__(name = 'UNet_model_simp_with_phys_layer')
        
        self.gamma = gamma
        
        # Downsamples Input
        if with_phys_layer:
            if with_learn_phys:
                self.Physical_Layer = Fast_Scan_Downsample(init_alpha = init_alpha, growth_rate = growth_rate, alpha_schedule = alpha_schedule)
            else:
                self.Physical_Layer = Uniform_Downsample(sparsity = sparsity, downsample_axis = downsample_axis)
                self.gamma = 0
        else:
            self.Physical_Layer = keras.activations.linear # Identity Activation
            self.gamma = 0
        
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
        self.sparsity = sparsity
        self.loss_func = self.model_loss()
        self.loss = self.model_loss()
        self.total_loss = self.model_loss()
    
    def call(self, input, training=None):
        #shortcut1_1 = input
        #print('input shape = '+str(input.shape))
        shortcut1_1 = self.Physical_Layer(input)
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
            #if self.Physical_Layer.count % 2 == 0:
            #    print(mask_num)
            #    print(mask_num - num_optim)
            # Full loss
            if self.gamma > 0:
                total_mask_num = K.count_params(self.Physical_Layer.mask)
                #print(total_mask_num)
                num_optim = (1 - self.sparsity)*total_mask_num
                #print(num_optim)
                mask_num = tf.reduce_sum(self.Physical_Layer.mask)
                loss = recon_loss + self.gamma*tf.math.abs(mask_num - num_optim)#**2
            else:
                loss = recon_loss
            return loss
        total_loss.__name__ = "total_loss"
        return total_loss
    def get_config(self):
        base_config = super(UNet_model_simp_with_phys_layer, self).get_config()
        base_config['gamma'] = self.gamma
        base_config['sparsity'] = self.sparsity
        base_config['loss_func'] = self.loss_func
        base_config['loss'] = self.loss
        base_config['total_loss'] = self.total_loss
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
    
class Fast_Scan_Downsample(tf.keras.layers.Layer):
    def __init__(self, init_alpha, growth_rate, alpha_schedule):
        super(Fast_Scan_Downsample, self).__init__(name = 'Fast_Scan_Downsample')
        self.init_alpha = init_alpha
        self.growth_rate = growth_rate
        self.alpha_schedule = alpha_schedule
    def build(self, input_shape):
        self.mask = None #tf.compat.v1.placeholder(tf.float32, shape=input_shape)
        self.weight_matrix = None #tf.compat.v1.placeholder(tf.float32, shape=input_shape)
        self.count = 0
        self.flatten = tf.keras.layers.Flatten()
        self.W = self.add_weight(name = 'W',
                                 shape=input_shape[1:],
                                 initializer=tf.keras.initializers.glorot_normal,#'random_normal',
                                 trainable=True)
        self.activation = tf.keras.layers.Activation('sigmoid') #tf.keras.layers.Activation('softmax')
        super(Fast_Scan_Downsample, self).build(input_shape)
    def call(self, input):
        if self.count > 0:
            alpha = self.alpha_growth_func(self.init_alpha, self.count, self.growth_rate, self.alpha_schedule)
        else:
            alpha = self.init_alpha
        alpha = self.alpha_growth_func(self.init_alpha, self.count, self.growth_rate, self.alpha_schedule)
        #print('alpha = ' + str(alpha))
        self.count += 1
        self.weight_matrix = tf.math.scalar_mul(tf.convert_to_tensor(alpha), self.W)
        self.mask = self.activation(self.weight_matrix)
        out = tf.math.multiply(input, self.mask)
        return out
    def alpha_growth_func(self, init_alpha, count, growth_rate, alpha_schedule):
        if count%alpha_schedule == 0:
            out = (count*growth_rate)**2+init_alpha
        else:
            new_count = count
            maxIter = 200
            iter = 0
            while new_count%alpha_schedule!=0 and iter < maxIter:
                new_count -= 1
                iter += 1
            out = (new_count*growth_rate)**2+init_alpha
        return out
    def get_config(self):
        base_config = super(Fast_Scan_Downsample, self).get_config()
        base_config['init_alpha'] = self.init_alpha
        base_config['growth_rate'] = self.growth_rate
        base_config['alpha_schedule'] = self.alpha_schedule
        return base_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class Uniform_Downsample(tf.keras.layers.Layer):
    def __init__(self, sparsity, downsample_axis):
        super(Uniform_Downsample, self).__init__(name = 'Uniform_Downsample')
        self.downsample_axis = downsample_axis
        self.sparsity = sparsity
        self.passthrough = False
    def build(self, input_shape):
        self.mask = None #tf.compat.v1.placeholder(tf.float32, shape=input_shape)
        self.W = tf.zeros(shape=input_shape[1:], dtype = tf.float32).numpy()
        if self.downsample_axis == 'both':
            downsample_ratio = int(round(1/(1 - np.sqrt(self.sparsity))))
        else:
            downsample_ratio = int(round(1/(1 - self.sparsity)))
        self.downsample_ratio = downsample_ratio
        batch_dim = input_shape[0]
        x_dim = input_shape[1]
        y_dim = input_shape[2]
        self.mask = self.W.copy()
        rand = 0#int(tf.random.uniform(shape=[1], minval = 0, maxval = 5, dtype=tf.float32).numpy()[0])
        for i in range(x_dim):
            for j in range(y_dim):
                if self.downsample_axis == 'both':
                    if i%downsample_ratio==rand or j%downsample_ratio==rand:
                        self.mask[i,j,0] = 1
                elif self.downsample_axis == 'y':
                    if i%downsample_ratio==rand:
                        self.mask[i,j,0] = 1
                else:
                    if j%downsample_ratio==rand:
                        self.mask[i,j,0] = 1
        self.mask = tf.convert_to_tensor(self.mask, dtype = tf.float32)
        self.mask_numpy = self.mask.numpy()
        super(Uniform_Downsample, self).build(input_shape)
    def call(self, input):
        if self.passthrough == False:
            if tf.executing_eagerly():
                rand_shift = K.cast(tf.random.uniform(shape=[1], minval = 0, 
                                                      maxval = self.downsample_ratio, 
                                                      dtype=tf.float32), tf.int32)[0]
                self.mask = tf.roll(self.mask, shift = rand_shift, axis = 1)
            out = tf.math.multiply(input, self.mask)
        elif self.passthrough:
            out = input
        return out
    def get_config(self):
        base_config = super(Uniform_Downsample, self).get_config()
        base_config['downsample_axis'] = self.downsample_axis
        base_config[''] = self.sparsity
        return base_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
# class Uniform_Downsample(tf.keras.layers.Layer):
#     def __init__(self, sparsity, downsample_axis):
#         super(Uniform_Downsample, self).__init__(name = 'Uniform_Downsample')
#         self.downsample_axis = downsample_axis
#         self.sparsity = sparsity
#     def build(self, input_shape):
#         self.mask = None #tf.compat.v1.placeholder(tf.float32, shape=input_shape)
#         self.W = tf.zeros(shape=input_shape[1:], dtype = tf.float32).numpy()
#         if self.downsample_axis == 'both':
#             downsample_ratio = int(round(1/(1 - np.sqrt(self.sparsity))))
#         else:
#             downsample_ratio = int(round(1/(1 - self.sparsity)))
#         batch_dim = input_shape[0]
#         x_dim = input_shape[1]
#         y_dim = input_shape[2]
#         self.mask = self.W.copy()
#         for i in range(x_dim):
#             for j in range(y_dim):
#                 if self.downsample_axis == 'both':
#                     if i%downsample_ratio==0 or j%downsample_ratio==0:
#                         self.mask[i,j,0] = 1
#                 elif self.downsample_axis == 'y':
#                     if i%downsample_ratio==0:
#                         self.mask[i,j,0] = 1
#                 else:
#                     if j%downsample_ratio==0:
#                         self.mask[i,j,0] = 1
#         self.mask = tf.convert_to_tensor(self.mask, dtype = tf.float32)
#         self.mask_numpy = self.mask.numpy()
#         super(Uniform_Downsample, self).build(input_shape)
#     def call(self, input):
#         out = tf.math.multiply(input, self.mask)
#         return out
#     def get_config(self):
#         base_config = super(Uniform_Downsample, self).get_config()
#         base_config['downsample_axis'] = self.downsample_axis
#         base_config[''] = self.sparsity
#         return base_config
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    
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
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
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
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
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
    