# Customary Imports:
import tensorflow as tf
assert '2.' in tf.__version__  # make sure you're using tf 2.0
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import skimage
import cv2 as cv
import os
import datetime
import scipy
from skimage.morphology import reconstruction
from skimage import exposure
import scipy.io as sio
import h5py
import random
import shutil
import PIL
import imageio
import pydot 
import graphviz
import plotly.graph_objects as go
import preprocess_crop
from pathlib import Path
from tensorflow.keras import backend as K
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
#from keras.utils import CustomObjectScope
from mpl_toolkits.mplot3d import Axes3D

##################################################################################################################################
'''
STANDARDIZE DIR UTILS:
'''
##################################################################################################################################
# Using RGB Channels to carry fully-sampled, downsampled, and downsampling mask of grayscale data:
def pad_img_and_add_down_channel(array=np.array(0), downsample_axis = 'x', downsample_ratio = [0,2], shape=[128,128]):
    '''
    This function takes in an image and outputs a three channel image, where the first channel
    is the fully-sampled image, the second sample is a downsampled version of this image, and
    the third channel contains the downsampling binary mask
    '''
    if len(array.shape) != 0:
        if len(shape)>2:
            shape = shape[0:2]
        if len(array.shape) > 2:
            array = np.mean(array, axis = 2)
        array = array/np.max(array)
        down_image = np.array(array, dtype = np.float32)
        mask = np.ones(array.shape)
        #print(full_image.shape)
        if downsample_ratio[0] == 1:
            downsample_ratio[0] = 0
        elif downsample_ratio[1] == 1:
            downsample_ratio[1] = 0
        if downsample_axis == 'x':
            downsample_ratio = downsample_ratio[1]
            for j in range(array.shape[1]):
                if j%downsample_ratio!=0:
                    mask[:, j] = 0
        elif downsample_axis == 'y':
            downsample_ratio = downsample_ratio[0]
            for i in range(array.shape[0]):
                if i%downsampling_ratio[1]!=0:
                    mask[i, :] = 0
        elif downsample_axis == 'both':
            downsample_ratio_j = downsample_ratio[1]
            downsample_ratio_i = downsample_ratio[0]
            if downsample_ratio_j > 0:
                for j in range(array.shape[1]):
                    if j%downsample_ratio[1]!=0:
                        mask[:, j] = 0
            if downsample_ratio_i > 0:
                for i in range(array.shape[0]):
                    if i%downsample_ratio[0]!=0:
                        mask[i, :] = 0
        down_image = np.multiply(mask, down_image)
        full_i_shape = array.shape[0]
        full_j_shape = array.shape[1]
        if full_i_shape%shape[0] != 0:
            i_left = full_i_shape%shape[0]
            i_pad = (shape[0] - i_left)//2
            rest_i = (shape[0] - i_left)%2
        else:
            i_left = 0
            i_pad = 0
            rest_i = 0
        if full_j_shape%shape[1] != 0:
            j_left = full_j_shape%shape[1]
            j_pad = (shape[1] - j_left)//2
            rest_j = (shape[1] - j_left)%2
        else:
            j_left = 0
            j_pad = 0
            rest_j = 0

        #print('i_left = '+str(i_left))
        #print('j_left = '+str(j_left))
        #print('i_pad = '+str(i_pad))
        #print('j_pad = '+str(j_pad))
        #print('rest_i = '+str(rest_i))
        #print('rest_j = '+str(rest_j))
        full_image = np.zeros((full_i_shape, full_j_shape, 3), dtype = np.float32)
        full_image[...,0] = array # Target Array
        full_image[...,1] = down_image # Downsampled Array
        full_image[...,2] = mask # Mask Array - for dispay
        pad_image = np.pad(full_image, [(i_pad, ), (j_pad, ), (0,)], mode='constant', constant_values = 0)
        padded_multi_chan_image = np.pad(pad_image, [(0, rest_i), (0, rest_j), (0, 0)], mode='constant', constant_values = 0)
    else:
        padded_multi_chan_image = np.array(0)
    return padded_multi_chan_image

output = pad_img_and_add_down_channel()

def standardize_dir(input_dir = 'data/train/input', downsample_axis = 'x', downsample_ratio = [0,2], 
                      standard_shape = (128, 128, 1), file_format = '.tif'):
    '''
    This function loops through an input directory and converts each file according to the
    function "pad_img_and_add_down_channel." The modified image is then saved into the 
    input directory and the original file is deleted. 
    '''
    file_list = os.listdir(input_dir)
    for file in file_list:
        # Load Image
        filename = os.fsdecode(file)
        filepath = os.path.join(input_dir, filename)
        if filepath.endswith('.npy'):
            array = np.load(filepath)
        else:
            array = imageio.imread(filepath)
            array = np.array(array)
        temp = pad_img_and_add_down_channel(array, downsample_axis = downsample_axis, 
                                            downsample_ratio = downsample_ratio, shape = standard_shape)
        # Save Image
        if file_format == '.npy':
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = Path(os.path.abspath(new_filepath)+ '_pad')
            new_filepath = new_filepath.with_suffix(file_format)
            np.save(new_filepath, temp, allow_pickle=True, fix_imports=True)
        else:
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = Path(os.path.abspath(new_filepath)+ '_pad')
            new_filepath = new_filepath.with_suffix(file_format)
            imageio.imwrite(new_filepath, temp)
        os.remove(filepath)
    return len(os.listdir(input_dir))