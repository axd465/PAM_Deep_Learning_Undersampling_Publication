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
MODEL HISTORY FUNCTIONS:
'''
def load_history_from_saved(directory):
    direct = os.path.join(os.getcwd(), directory)
    filepath = os.path.join(direct, 'history.npz')
    array = np.load(filepath, allow_pickle=True, fix_imports=True)
    return array
def show_history(hist, offset=50):
    data1 = history.history['loss'][offset:]
    epochs = range(offset,len(data1)+offset)
    plt.plot(epochs, data1)
    data2 = history.history['val_loss'][offset:]
    plt.plot(epochs, data2)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc = "upper left")
    #plt.xticks(epochs)
    plt.show()
    
    data1 = history.history['mean_absolute_error'][offset:]
    epochs = range(offset,len(data1)+offset)
    plt.plot(epochs, data1)
    data2 = history.history['val_mean_absolute_error'][offset:]
    plt.plot(epochs, data2)
    plt.title('Mean Absolute Error')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc = "upper left")
    #plt.xticks(epochs)
    plt.show()
    
    data1 = history.history['mean_squared_error'][offset:]
    epochs = range(offset,len(data1)+offset)
    plt.plot(epochs, data1)
    data2 = history.history['val_mean_squared_error'][offset:]
    plt.plot(epochs, data2)
    plt.title('Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc = "upper left")
    #plt.xticks(epochs)
    plt.show()
    
    data1 = history.history['KLDivergence'][offset:]
    epochs = range(offset,len(data1)+offset)
    plt.plot(epochs, data1)
    data2 = history.history['val_KLDivergence'][offset:]
    plt.plot(epochs, data2)
    plt.title('KLDivergence')
    plt.ylabel('KLDivergence')
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc = "upper left")
    #plt.xticks(epochs)
    plt.show()
    
    data1 = history.history['PSNR'][offset:]
    epochs = range(offset,len(data1)+offset)
    plt.plot(epochs, data1)
    data2 = history.history['val_PSNR'][offset:]
    plt.plot(epochs, data2)
    plt.title('Peak Signal-to-Noise')
    plt.ylabel('PSNR')
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc = "upper left")
    #plt.xticks(epochs)
    plt.show()
    
    data1 = history.history['SSIM'][offset:]
    epochs = range(offset,len(data1)+offset)
    plt.plot(epochs, data1)
    data2 = history.history['val_SSIM'][offset:]
    plt.plot(epochs, data2)
    plt.title('Structural Similarity Index')
    plt.ylabel('SSIM')
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc = "upper left")
    #plt.xticks(epochs)
    plt.show()