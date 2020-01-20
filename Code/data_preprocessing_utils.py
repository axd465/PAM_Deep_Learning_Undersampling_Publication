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
DATA PREPROCESSING UTILS:
'''
##################################################################################################################################
# Converting MAP Files:
def convert_MAP(directory, output_directory, min_shape, file_format = '.npy', search_keys = None, dtype = np.float32):
    '''
    This program loops through given raw_data directory
    and converts .mat files to .npy files
    '''
    new_dir = os.path.join(os.getcwd(), output_directory)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    else:
        shutil.rmtree(new_dir)
        os.mkdir(new_dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mat"): 
            #print(os.path.join(directory, filename))
            filepath = os.path.join(directory, filename)
            array_dict = {}
            try:
                f = h5py.File(filepath, 'r')
            except:
                f = sio.loadmat(filepath)
            for k, v in f.items():
                array_dict[k] = np.array(v, dtype = np.float32)
            # As we only need image info from dict (the last key) we do this
            if search_keys == None:
                search_keys = 'map' # out of struct of .mat files want "map"
                filtered_dict = dict(filter(lambda item: search_keys in item[0], array_dict.items()))
            else:
                filtered_dict = {}
                for i in range(len(search_keys)):
                    search_key = search_keys[i]
                    if search_key in array_dict:
                        filtered_dict[search_key] = array_dict[search_key]
            if len(filtered_dict) == 0:
                print('No Data to Meet Search Key Requirements: Datapoint Rejected -> ' + filepath)
            else:
                #print(list(array_dict.keys()))
                #print(filtered_dict)
                arrays = []
                for k, v in filtered_dict.items():
                    temp = np.transpose(v.astype(np.float32))
                    # To normalize data between [-1,1], use -> arrays = arrays/(np.max(arrays)/2) - 1
                    # To normalize data between [0,1], use -> arrays = arrays/(np.max(arrays))
                    # To normalize data between [0,255], 
                    #     use -> arrays = (arrays/(np.max(arrays))*255).astype(np.uint8)
                    temp = temp/(np.max(temp))
                    arrays.append(temp)
                for i in range(len(arrays)):
                    if len(arrays[i].shape) > 2:
                        #print(arrays[i].shape)
                        arrays[i] = np.mean(arrays[i], axis = 2)

                for i in range(len(arrays)):
                    new_dir_filepath = os.path.join(new_dir, filename.strip('.mat') 
                                                    + '_index'+str(i) + file_format)
                    array = arrays[i]
                    if array.shape[0] >= min_shape[0] and array.shape[1] >= min_shape[1]:
                        if file_format == '.npy':
                            np.save(new_dir_filepath, array, allow_pickle=True, fix_imports=True)
                        else:
                            imageio.imwrite(new_dir_filepath, array)
                    elif i == 0:
                        print('Min Size Not Met: Datapoint Rejected -> ' + filepath)
    return os.path.join(os.getcwd(), output_directory)

##################################################################################################################################
# Data Cleaning Procedures:
def data_clean_func(image = np.array(0)):
    if len(image.shape) != 0:
        #print(len(np.unique(image)))
        #clean_image = image
        '''
        plt.hist(image)
        plt.show()
        '''
        '''
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.show()
        '''
        threshold = 0.85
        default_fill = 0.0
        frac_of_high_clip = 1/9
        image[image > threshold] = default_fill
        image[image < frac_of_high_clip*(1.0-threshold)] = default_fill
        '''
        plt.imshow(image, cmap='gray')
        plt.title('After Clipping')
        plt.show()
        '''
        image = scipy.ndimage.median_filter(image, size=(4, 4))
        '''
        plt.imshow(image, cmap='gray')
        plt.title('After Median Filter')
        plt.show()
        '''
        image = skimage.filters.gaussian(image, sigma=0.01, output=None, mode='reflect', preserve_range=True)
        ####################################################################
        # Added to ensure negligible loss when converting to int16 
        # within exposure.equalize_adapthist
        image = (image/np.max(image)*(2**16)).astype(np.uint16)
        # A "Monkey Patch" could possibly be used as a cleaner solution, 
        # but would be more involved than is necessary for my application
        ####################################################################
        image = exposure.equalize_adapthist(image,kernel_size=image.shape[0]//8, clip_limit=0.005, nbins=2**13)
        image = image.astype(np.float64)
        '''
        plt.imshow(image, cmap='gray')
        plt.title('After Local Adapt Hist')
        plt.show()
        '''
        image = scipy.ndimage.median_filter(image, size=(3, 1))
        image = scipy.ndimage.median_filter(image, size=(1, 3))
        image = skimage.filters.gaussian(image, sigma=0.1, output=None, mode='reflect', preserve_range=True)
        image = exposure.rescale_intensity(image, in_range='image', out_range=(0.0,1.0))
        '''
        plt.imshow(image, cmap='gray')
        plt.title('Final Image')
        plt.show()
        '''
        '''
        plt.hist(image)
        plt.show()
        '''
        clean_image = image.astype(np.float32)
    else:
        clean_image = image
    return clean_image

output = data_clean_func()

def data_cleaning(input_dir = 'converted_data', output_dir_name = 'cleaned_data',
                  output_file_format ='.npy', delete_previous = True):
    '''
     This program seeks to remove some noise from the data
     and make the underlying vessel structure more prominent
     Input: input_dir -> directory that holds data to be cleaned
            output_dir -> directory to hold cleaned data
     Output: None
    '''
    file_list = os.listdir(input_dir)
    clean_dir = os.path.join(os.getcwd(), output_dir_name)
    if not os.path.exists(clean_dir):
        os.mkdir(clean_dir)
    elif delete_previous == True:
        shutil.rmtree(clean_dir)
        os.mkdir(clean_dir)
    for file in file_list:
        filename = os.fsdecode(file)
        filepath = os.path.join(input_dir, filename)
        if filepath.endswith('.npy'):
            array = np.load(filepath)
        else:
            array = imageio.imread(filepath)
            
        # Defined data clean function above:
        array = data_clean_func(array)
    
        new_filepath = os.path.join(clean_dir, filename)
        if output_file_format == '.npy':
            new_filepath = Path(new_filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(output_file_format)
            np.save(new_filepath, array, allow_pickle=True, fix_imports=True)
        else:
            new_filepath = Path(new_filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(output_file_format)
            imageio.imwrite(new_filepath, array)
    return  

    
##################################################################################################################################
# Data Seperation / Validation Split Procedures:
def data_seperation(input_dir, dataset_percentages, 
                    delete_previous = False, file_format = '.npy', 
                    scale = 1):
    '''
    Takes numpy array and creates data folder with seperate sections
    for training, validation, and testing according to given percentages
    Input: numpy dir -> contains file path to data folder of numpy files
           dataset_percentages -> (% train, % test) such that % train + % test = 100
           OR
           dataset_percentages -> (% train, % val, % test) such that % train + % val + % test = 100
    Output: new folders for training and testing or training/validation/testing
    '''
    
    # If just train and test
    if len(dataset_percentages) == 2:
        # Making Main data folder
        new_dir = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        
        # Making train subfolder
        train_dir = os.path.join(new_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
            train_dir = os.path.join(train_dir, 'input')
            os.mkdir(train_dir)
        elif delete_previous == True:
            shutil.rmtree(train_dir)
            os.mkdir(train_dir)
            train_dir = os.path.join(train_dir, 'input')
            os.mkdir(train_dir)
        
        # Making test subfolder
        test_dir = os.path.join(new_dir, 'test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
            test_dir = os.path.join(test_dir, 'input')
            os.mkdir(test_dir)
        elif delete_previous == True:
            shutil.rmtree(test_dir)
            os.mkdir(test_dir)
            test_dir = os.path.join(test_dir, 'input')
            os.mkdir(test_dir)


        file_list = os.listdir(input_dir)
        total_num_imgs = len(file_list)
        train_percent = dataset_percentages[0]
        test_percent = dataset_percentages[1]
        valid_inputs = (train_percent >= test_percent and train_percent <= 100 and
                        test_percent <= 100 and train_percent > 0 and test_percent > 0 and
                        train_percent + test_percent == 100)
        if valid_inputs:
            num_train = int(round(total_num_imgs * train_percent//100))
        else:
            num_train = int(round(total_num_imgs * 0.9))
            print('ERROR: Please input valid percentages for dataset division')
            print('In place of valid input the ratio 90% train, 10% test was used')
        
        index = 0
        random.shuffle(file_list)
        for file in file_list:
            filename = os.fsdecode(file)
            filepath = os.path.join(input_dir, filename)
            # Loads File
            if filepath.endswith('.npy'):
                array = np.load(filepath)
                array = array/np.max(array)*scale
            else:
                array = imageio.imread(filepath)
                array = array/np.max(array)*scale
            if index < num_train:
                new_filepath = os.path.join(train_dir, filename)
            else:
                new_filepath = os.path.join(test_dir, filename)
            # Saves File
            if file_format == '.npy':
                new_filepath = Path(new_filepath)
                new_filepath = new_filepath.with_suffix('')
                new_filepath = new_filepath.with_suffix(file_format)
                np.save(new_filepath, array, allow_pickle=True, fix_imports=True)
            else:
                new_filepath = Path(new_filepath)
                new_filepath = new_filepath.with_suffix('')
                new_filepath = new_filepath.with_suffix(file_format)
                imageio.imwrite(new_filepath, array)
            index += 1
        return train_dir, test_dir
    # If train, val, and test
    elif len(dataset_percentages) == 3:
        # Making Main data folder
        new_dir = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
            
        # Making train subfolder
        train_dir = os.path.join(new_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
            train_dir = os.path.join(train_dir, 'input')
            os.mkdir(train_dir)
        elif delete_previous == True:
            shutil.rmtree(train_dir)
            os.mkdir(train_dir)
            train_dir = os.path.join(train_dir, 'input')
            os.mkdir(train_dir)
        
        # Making val subfolder
        val_dir = os.path.join(new_dir, 'val')
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
            val_dir = os.path.join(val_dir, 'input')
            os.mkdir(val_dir)
        elif delete_previous == True:
            shutil.rmtree(val_dir)
            os.mkdir(val_dir)
            val_dir = os.path.join(val_dir, 'input')
            os.mkdir(val_dir)
        
        # Making test subfolder
        test_dir = os.path.join(new_dir, 'test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
            test_dir = os.path.join(test_dir, 'input')
            os.mkdir(test_dir)
        elif delete_previous == True:
            shutil.rmtree(test_dir)
            os.mkdir(test_dir)
            test_dir = os.path.join(test_dir, 'input')
            os.mkdir(test_dir)
            
        file_list = os.listdir(input_dir)
        total_num_imgs = len(file_list)
        train_percent = dataset_percentages[0]
        val_percent = dataset_percentages[1]
        test_percent = dataset_percentages[2]
        valid_inputs = (train_percent >= test_percent and train_percent >= val_percent 
                        and train_percent <= 100 and val_percent <= 100 and test_percent <= 100
                        and train_percent > 0 and val_percent > 0 and test_percent > 0 and
                        train_percent + val_percent + test_percent == 100)
        if valid_inputs:
            num_train = int(round(total_num_imgs * train_percent//100))
            num_val = int(round(total_num_imgs * val_percent//100))
        else:
            num_train = int(round(total_num_imgs * 0.9))
            num_val = int(round((total_num_imgs - num_train)/2))
            print('ERROR: Please input valid percentages for dataset division')
            print('In place of a valid input the ratio 90% train, 5% val, 5% test was used')
        
        index = 0
        random.shuffle(file_list)
        for file in file_list:
            filename = os.fsdecode(file)
            filepath = os.path.join(input_dir, filename)
            # Loads File
            if filepath.endswith('.npy'):
                array = np.load(filepath)
                array = array/np.max(array)*scale
            else:
                array = imageio.imread(filepath)
                array = array/np.max(array)*scale
            if index < num_train:
                new_filepath = os.path.join(train_dir, filename)
            elif index <= num_train + num_val:
                new_filepath = os.path.join(val_dir, filename)
            else:
                new_filepath = os.path.join(test_dir, filename)
            # Saves File
            if file_format == '.npy':
                new_filepath = Path(new_filepath)
                new_filepath = new_filepath.with_suffix('')
                new_filepath = new_filepath.with_suffix(file_format)
                np.save(new_filepath, array, allow_pickle=True, fix_imports=True)
            else:
                new_filepath = Path(new_filepath)
                new_filepath = new_filepath.with_suffix('')
                new_filepath = new_filepath.with_suffix(file_format)
                imageio.imwrite(new_filepath, array)
            index += 1
        return train_dir, val_dir, test_dir
    else:
        print('ERROR: Please divide into train/test or train/val/test')
        return None