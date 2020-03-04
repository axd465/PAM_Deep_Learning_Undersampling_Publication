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
PATCHWORK ALGORITHM FUNCTIONS:
'''
def expand_image(down_image = np.array(0), downsampling_ratio = [2, 5], downsampling_axis = 'x', output_shape = None):
    '''
    This function expands a given image according to inputted ratio by resizing with fill_value = 0.
    By performing this operation the function seeks to place zero value pixels where downsampling
    occured within the image during image acquisition.
    '''
    if len(down_image.shape) != 0:
        if downsampling_ratio[0]==0:
            downsampling_ratio[0]=1
        if downsampling_ratio[1]==0:
            downsampling_ratio[1]=1
        i_shape = down_image.shape[0]
        j_shape = down_image.shape[1]
        if output_shape == None:
            if downsampling_axis == 'x':
                i_shape_desired = i_shape
                j_shape_desired = int(j_shape * downsampling_ratio[1])
            elif downsampling_axis == 'y':
                i_shape_desired = int(i_shape * downsampling_ratio[0])
                j_shape_desired = j_shape
            elif downsampling_axis == 'both':
                i_shape_desired = int(i_shape * downsampling_ratio[0])
                j_shape_desired = int(j_shape * downsampling_ratio[1])
            else:
                print('ERROR: Please input x or y as downsampling axis')
        else:
            if output_shape[0] >= i_shape and output_shape[1] >= j_shape:
                i_shape_desired = output_shape[0]
                j_shape_desired = output_shape[1]
            else: 
                i_shape_desired = i_shape
                j_shape_desired = j_shape
        full_image = np.zeros((i_shape_desired, j_shape_desired), dtype = np.float32)
        #print(full_image.shape)
        decay_factor = 4
        if downsampling_axis == 'x':
            count = 0
            for j in range(j_shape_desired):
                if j%downsampling_ratio[1]==0 or j>=(j_shape_desired-downsampling_ratio[1]//2):
                    if j%downsampling_ratio[1]==0:
                        full_image[:, j] = down_image[:, count]
                        count += 1
                    else:
                        j_inter = j_shape_desired - j
                        full_image[:, j] = (j_inter/decay_factor)*down_image[:, count-1]    
        elif downsampling_axis == 'y':
            count = 0
            for i in range(i_shape_desired):
                if i%downsampling_ratio[0]==0 or i>=(i_shape_desired-downsampling_ratio[0]//2):
                    if i%downsampling_ratio[0]==0:
                        full_image[i, :] = down_image[count, :]
                        count += 1
                    else:
                        i_inter = i_shape_desired - i
                        full_image[i, :] = (i_inter/decay_factor)*down_image[count-1, :]                             
        elif downsampling_axis == 'both':
            i_count = 0
            for i in range(i_shape_desired):
                j_count = 0
                for j in range(j_shape_desired):
                    if ((i%downsampling_ratio[0]==0 and j%downsampling_ratio[1]==0) or 
                        (i>=(i_shape_desired-downsampling_ratio[0]//2) or j>=(j_shape_desired-downsampling_ratio[1]//2))):
                        #print('index = '+str(i) + ', ' + str(j))
                        if (i%downsampling_ratio[0]==0 and j%downsampling_ratio[1]==0):
                            full_image[i, j] = down_image[i_count, j_count]
                        elif i>=(i_shape_desired-downsampling_ratio[0]//2) and j%downsampling_ratio[1]==0:
                            i_inter = i_shape_desired - i
                            full_image[i, j] = (i_inter/decay_factor)*down_image[i_count-1, j_count]
                        elif  j>=(j_shape_desired-downsampling_ratio[1]//2) and i%downsampling_ratio[0]==0:
                            j_inter = j_shape_desired - j
                            full_image[i, j] = (j_inter/decay_factor)*down_image[i_count, j_count-1]
                        elif i>=(i_shape_desired-downsampling_ratio[0]//2) and j>=(j_shape_desired-downsampling_ratio[1]//2):
                            i_inter = i_shape_desired - i
                            j_inter = j_shape_desired - j
                            rest_vec = np.sqrt(i_inter**2 + j_inter**2)
                            full_image[i, j] = (rest_vec/decay_factor)*down_image[i_count-1, j_count-1]
                        #print('count = '+str(i_count) + ', ' + str(j_count))
                    if j%downsampling_ratio[1]==0:
                        j_count += 1
                if i%downsampling_ratio[0]==0:
                    i_count += 1
        else:
            print('ERROR: Please input x or y as downsampling axis')
        full_image = exposure.rescale_intensity(full_image, in_range='image', out_range=(0.0,1.0))
    else:
        full_image = down_image
    return full_image

def fix_boundaries(orig_img=np.array(0), patch_img=np.array(0), model=None, i_count=0, j_count=0, pad_image_shape=(128,128), model_input_shape = (128,128), bound_buff = 4):
    '''
    This function augments the patchwork algorithm and makes sure the seams between the patches do not have any undue edge
    distortion.
    '''
    if len(orig_img.shape) != 0 and len(patch_img.shape) != 0 and model != None:
        img = patch_img
        if i_count == 1:
            if j_count > 1:
                for j in range(model_input_shape[1]//2,pad_image_shape[1]-model_input_shape[1]//2,model_input_shape[1]):
                    patch = orig_img[:,j:j+model_input_shape[1]]
                    #print(patch.shape)
                    patch = patch[..., None]
                    pred = model.predict(patch[None], batch_size = 1)
                    pred = pred[0,:,:,0]

                    #plt.imshow(pred, cmap='gray')
                    #plt.show()
                    mid_patch = j+model_input_shape[1]//2
                    img[:,mid_patch-bound_buff:mid_patch+bound_buff] = pred[:, pred.shape[1]//2-bound_buff:pred.shape[1]//2+bound_buff]
        else:
            for i in range(0,pad_image_shape[0],model_input_shape[0]):
                for j in range(model_input_shape[1]//2,pad_image_shape[1]-model_input_shape[1]//2,model_input_shape[1]):
                    patch = orig_img[i:i+model_input_shape[0],j:j+model_input_shape[1]]
                    patch = patch[..., None]
                    pred = model.predict(patch[None], batch_size = 1)
                    pred = pred[0,:,:,0]

                    #plt.imshow(pred, cmap='gray')
                    #plt.show()
                    mid_patch = j+model_input_shape[1]//2
                    #plt.imshow(img[i:i+model_input_shape[1],mid_patch-bound_buff:mid_patch+bound_buff], cmap='gray')
                    #plt.show()
                    #'''
                    # TO NOT SHOW PATCH PATTERN
                    img[i:i+model_input_shape[0],
                        mid_patch-bound_buff:mid_patch+bound_buff] = pred[:, pred.shape[1]//2-bound_buff:pred.shape[1]//2+bound_buff]
                    #'''
                    '''
                    # TO SHOW PATCH PATTERN
                    img[i:i+model_input_shape[0],
                        mid_patch-bound_buff:mid_patch+bound_buff] = np.ones(pred[:, pred.shape[1]//2-bound_buff:
                                                                                  pred.shape[1]//2+bound_buff].shape)
                    '''
                    #plt.imshow(img[i:i+model_input_shape[0],mid_patch-bound_buff:mid_patch+bound_buff], cmap='gray')
                    #plt.show()
            for j in range(0,pad_image_shape[1],model_input_shape[1]):
                for i in range(model_input_shape[0]//2,pad_image_shape[0]-model_input_shape[0]//2,model_input_shape[0]):
                    patch = orig_img[i:i+model_input_shape[0],j:j+model_input_shape[1]]
                    patch = patch[..., None]
                    pred = model.predict(patch[None], batch_size = 1)
                    pred = pred[0,:,:,0]

                    #plt.imshow(pred, cmap='gray')
                    #plt.show()
                    mid_patch = i+model_input_shape[0]//2
                    #'''
                    # TO NOT SHOW PATCH PATTERN
                    img[mid_patch-bound_buff:mid_patch+bound_buff, 
                        j:j+model_input_shape[1]] = pred[pred.shape[0]//2-bound_buff:pred.shape[0]//2+bound_buff,:]
                    #'''
                    '''
                    # TO SHOW PATCH PATTERN
                    img[mid_patch-bound_buff:mid_patch+bound_buff, 
                        j:j+model_input_shape[1]] = np.zeros(pred[pred.shape[0]//2-bound_buff:
                                                                  pred.shape[0]//2+bound_buff,:].shape)
                    '''
            # Cover Overlap
            pad = bound_buff//4+1
            bound_buff += pad
            for i in range(model_input_shape[0]//2,pad_image_shape[0]-model_input_shape[0]//2,model_input_shape[0]):
                for j in range(model_input_shape[1]//2,pad_image_shape[1]-model_input_shape[1]//2,model_input_shape[1]):
                    patch = orig_img[i:i+model_input_shape[0],j:j+model_input_shape[1]]
                    patch = patch[..., None]
                    pred = model.predict(patch[None], batch_size = 1)
                    pred = pred[0,:,:,0]

                    #plt.imshow(pred, cmap='gray')
                    #plt.show()
                    mid_patch_i = i+model_input_shape[0]//2
                    mid_patch_j = j+model_input_shape[1]//2
                    #plt.imshow(img[i:i+model_input_shape[1],mid_patch-bound_buff:mid_patch+bound_buff], cmap='gray')
                    #plt.show()
                    #'''
                    # TO NOT SHOW PATCH PATTERN
                    img[mid_patch_i-bound_buff:mid_patch_i+bound_buff,
                        mid_patch_j-bound_buff:mid_patch_j+bound_buff] = pred[pred.shape[0]//2-bound_buff:pred.shape[0]//2+bound_buff, 
                                                                              pred.shape[1]//2-bound_buff:pred.shape[1]//2+bound_buff]
                    #'''
                    '''
                    # TO SHOW PATCH PATTERN
                    img[mid_patch_i-bound_buff:mid_patch_i+bound_buff,
                        mid_patch_j-bound_buff:mid_patch_j+bound_buff] = 0.5*np.ones(pred[pred.shape[0]//2-bound_buff:pred.shape[0]//2+bound_buff, 
                                                                                          pred.shape[1]//2-bound_buff:pred.shape[1]//2+bound_buff].shape)
                    '''
                    #plt.imshow(img[i:i+model_input_shape[0],mid_patch-bound_buff:mid_patch+bound_buff], cmap='gray')
                    #plt.show()
    else:
        img = orig_img
    return img
def remove_peak(image, num_std = 4):
    MAX_STD = 10
    image  = exposure.rescale_intensity(image, in_range='image', out_range=(0.0,1.0))
    orig_image = image
    if num_std > MAX_STD:
        num_std = MAX_STD
    mean = np.mean(image)
    std = np.std(image)
    #print(mean)
    #print(std)
    step = -0.001
    for std_lev in np.arange(MAX_STD, num_std, step):
        threshold = mean + std_lev*std
        if threshold > 1:
            threshold = 1
        decay_factor = 0.1
        if len(image[image > threshold].shape) > 0:
            image[image > threshold] = np.mean(image[image > threshold])*(1 - decay_factor)
    image  = exposure.rescale_intensity(image, in_range='image', out_range=(0.0,1.0))
    '''
    # SHOW CLEANED IMAGE:
    figsize = (15,15)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image, cmap = 'gray')
    plt.title('Cleaned Latent Image')
    '''
    return image
##################################################################################################################################
'''
PATCHWORK ALGORITHM:
'''
output = expand_image()
output = fix_boundaries()
def apply_model_patchwork(model, down_image, downsampling_ratio = 2, downsampling_axis = 'x', 
                          shape_for_model = (128,128), buffer = 10, output_shape = None):
    '''
    This function expands the image and then performs the model patchwork algorithm. Patches
    of the image are extracted and processed by the given CNN model.
    '''
    # This Function Expands the Image and Then Performs Model Patchwork Algorithm
    STARTING_POINT = (0,0)
    i_shape = down_image.shape[0]
    j_shape = down_image.shape[1]
    full_image = expand_image(down_image, downsampling_ratio = downsampling_ratio, 
                              downsampling_axis = downsampling_axis, output_shape = output_shape)
    full_i_shape = full_image.shape[0]
    full_j_shape = full_image.shape[1]
    default_pad = np.max(shape_for_model)//2
    #default_pad = 0
    if full_i_shape%shape_for_model[0] != 0:
        i_left = full_i_shape%shape_for_model[0]
        i_pad = (shape_for_model[0] - i_left)//2
        rest_i = (shape_for_model[0] - i_left)%2
    else:
        i_left = 0
        i_pad = default_pad
        rest_i = 0
    if full_j_shape%shape_for_model[1] != 0:
        j_left = full_j_shape%shape_for_model[1]
        j_pad = (shape_for_model[1] - j_left)//2
        rest_j = (shape_for_model[1] - j_left)%2
    else:
        j_left = 0
        j_pad = default_pad
        rest_j = 0
    
    #print('i_left = '+str(i_left))
    #print('j_left = '+str(j_left))
    #print('i_pad = '+str(i_pad))
    #print('j_pad = '+str(j_pad))
    #print('rest_i = '+str(rest_i))
    #print('rest_j = '+str(rest_j))
    pad_image = np.pad(full_image, [(i_pad, ), (j_pad, )], mode='constant')
    full_pad_image = np.pad(pad_image, [(0, rest_i), (0, rest_j)], mode='constant')
    
    '''
    figsize = (15,15)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(full_pad_image, cmap = 'gray')
    plt.show()
    '''
    
    orig_image_j_start = j_pad #+ rest_j
    orig_image_i_start = i_pad #+ rest_i
    orig_image_j_end = orig_image_j_start + full_j_shape
    orig_image_i_end = orig_image_i_start + full_i_shape
    #print(orig_image_i_start)
    #print(orig_image_i_end)
    #print(orig_image_j_start)
    #print(orig_image_j_end)
    full_patch_image = np.ones(full_pad_image.shape)
    i_count = 0
    for i in range(0,full_pad_image.shape[0],shape_for_model[0]):
        j_count = 0
        batch = np.zeros((full_pad_image.shape[1]//shape_for_model[1], 
                          shape_for_model[0], shape_for_model[1]))
        #print(batch.shape)
        for j in range(0,full_pad_image.shape[1],shape_for_model[1]):
            patch = full_pad_image[i:i+shape_for_model[0], j:j+shape_for_model[1]]
            start_i_temp =i_count*shape_for_model[0]
            end_i_temp = (i_count+1)*shape_for_model[0]
            start_j_temp = j_count*shape_for_model[1]
            end_j_temp = (j_count+1)*shape_for_model[1]
            #full_patch_image[start_i_temp:end_i_temp, start_j_temp:end_j_temp] = patch
            #plt.imshow(patch, cmap='gray')
            #plt.show()
            patch = patch[..., None]
            pred = model.predict(patch[None], batch_size = 1)
            #plt.imshow(pred[0,:,:,0], cmap='gray')
            #plt.show()
            full_patch_image[start_i_temp:end_i_temp, start_j_temp:end_j_temp] = pred[0,:,:,0]
            j_count +=1
        i_count +=1
    #'''
    # To Fix Patch Boundaries With Second and Third Passes
    if i_count > 1 or j_count > 1:
        full_patch_image = fix_boundaries(full_pad_image, full_patch_image, model, i_count, j_count, pad_image_shape = full_pad_image.shape,
                                          model_input_shape = shape_for_model, bound_buff = buffer)
    #'''
    #start_i_temp = orig_image_i_start + i_count*shape_for_model[0]
    #end_i_temp = orig_image_i_start + (i_count+1)*shape_for_model[0]
    #start_j_temp = orig_image_j_start + j_count*shape_for_model[1]
    #end_j_temp = orig_image_j_start + (j_count+1)*shape_for_model[1]
    #print('start_i_temp = '+str(start_i_temp))
    #print('end_i_temp = '+str(end_i_temp))
    #print('start_j_temp = '+str(start_j_temp))
    #print('end_j_temp = '+str(end_j_temp))
    full_recon_image = full_patch_image[orig_image_i_start:orig_image_i_end,orig_image_j_start:orig_image_j_end]
    full_recon_image = exposure.rescale_intensity(full_recon_image, in_range='image', out_range=(0.0,1.0))
    '''
    # Post Processing to Remove Salt-Pepper Throwing Off Contrast
    full_recon_image = scipy.ndimage.median_filter(full_recon_image, footprint=((1,0,0),(0,1,0),(0,0,1)))
    full_recon_image = scipy.ndimage.median_filter(full_recon_image, footprint=((0,1,0),(0,1,0),(0,1,0)))
    full_recon_image = scipy.ndimage.median_filter(full_recon_image, footprint=((0,0,1),(0,1,0),(1,0,0)))
    full_recon_image = scipy.ndimage.median_filter(full_recon_image, footprint=((0,0,0),(1,1,1),(0,0,0)))
    '''
    #full_recon_image = exposure.equalize_hist(full_recon_image, nbins=2**(13))
    full_recon_image = exposure.rescale_intensity(full_recon_image, in_range='image', out_range=(0.0,1.0))
    return full_recon_image