# Customary Imports:
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
from skimage import exposure
from pathlib import Path
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Add, Input
from tensorflow.keras.layers import BatchNormalization, UpSampling2D, Concatenate, Conv2DTranspose

##################################################################################################################################
'''
DATA AUGMENTATION FUNCTIONS:
'''
##################################################################################################################################
def preprocess_function(image, prob = 0.1, max_shift = 0.10, lower = 1.0, upper = 1.25, seed=7):
    '''
    This function is the wrapper preprocess function to be used on all training data as a data
    augmentation step.
    '''
    img = image.copy()
    #img = add_rand_contrast(img, lower, upper, prob, seed=seed)
    img = add_gaussian_noise(img, std = 0.1, prob = prob, seed=seed)
    '''
    img = tf.keras.preprocessing.image.random_zoom(img,zoom_range=(0.8,2),row_axis=0,col_axis=1,channel_axis=2,
                                                   fill_mode='nearest',cval=0.0,interpolation_order=3)
    '''
    img = exposure.rescale_intensity(img, in_range = 'image', out_range = (0.0,1.0))
    #print(img.shape)
    return img
def preprocess_function_valtest(image):
    '''
    This function is the wrapper preprocess function to be used on all validation/test data.
    '''
    img = image.copy()
    img = exposure.rescale_intensity(img, in_range='image', out_range=(0.0,1.0))
    return img
def add_gaussian_noise(batch, mean_val = 0.0, std = 0.1, prob = 0.1, seed = None):
    '''
    This function introduces additive Gaussian Noise with a given mean and std, at
    a certain given probability.
    '''
    rand_var = tf.random.uniform(shape = [1], seed=seed).numpy()[0]
    batch_and_noise = batch
    if rand_var < prob:
        noise = tf.random.normal(shape=tf.shape(batch), mean=mean_val, 
                                 stddev=std, dtype=tf.float32, seed=seed)
        batch_and_noise = tf.math.add(batch,noise)
        batch_and_noise = batch_and_noise.numpy()
    return batch_and_noise
##################################################################################################################################
'''
UNUSED FUNCTIONS:
'''
##################################################################################################################################
def add_rand_bright_shift(batch, max_shift = 0.12, prob = 0.1, seed=None):
    '''
    Equivalent to adjust_brightness() using a delta randomly
    picked in the interval [-max_delta, max_delta) with a
    given probability that this function is performed on an image
    '''
    rand_var = tf.random.uniform(shape = [1], seed=seed).numpy()[0]
    batch_and_bright_shift = batch
    if rand_var < prob:
        batch_and_bright_shift = tf.image.random_brightness(image=batch, max_delta=max_shift, 
                                                            seed=seed)
        batch_and_bright_shift = batch_and_bright_shift.numpy()
    return batch_and_bright_shift
def add_rand_contrast(batch, lower = 0.2, upper = 1.8, prob = 0.1, seed=None):
    '''
    For each channel, this Op computes the mean of the image pixels in the channel 
    and then adjusts each component x of each pixel to (x - mean) * contrast_factor + mean
    with a given probability that this function is performed on an image
    '''
    rand_var = tf.random.uniform(shape = [1],seed=seed).numpy()[0]
    batch_and_rand_contrast = batch
    if rand_var < prob:
        batch_and_rand_contrast = tf.image.random_contrast(image=batch, lower=lower, 
                                                           upper=upper, seed=seed)
        batch_and_rand_contrast = batch_and_rand_contrast.numpy()
    return batch_and_rand_contrast
