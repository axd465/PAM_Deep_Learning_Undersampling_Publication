# Customary Imports:
import numpy as np
import matplotlib.pyplot as plt
import math
import string
import pandas as pd
import sklearn
import skimage
import cv2 as cv
import os
import re
import datetime
import scipy
from skimage.morphology import reconstruction
from skimage import exposure
import scipy.io as sio
import seaborn as sns
import h5py
import random
import shutil
import PIL
import imageio
import copy
import pydot 
import graphviz
import plotly.graph_objects as go
from pathlib import Path

# Added Import Statements:
import remove_pad_save_to_new_dir
##################################################################################################################################
'''
REMOVING PADDING FROM IMAGE AND SAVING IN OUTPUT DIR:
'''
def remove_padding(img):
    # Removing Image Border (if it exists)
    xmin=0
    xmax=img.shape[0]-1
    ymin=0
    ymax=img.shape[1]-1
    for xmin in range(img.shape[0]):
        if np.sum(img[xmin,:,0]) > 0.01:
            break
    for xmax in range(img.shape[0]-1, 0, -1):
        if np.sum(img[xmax,:,0]) > 0.01:
            break
    for ymin in range(img.shape[1]):
        if np.sum(img[:,ymin,0]) > 0.01:
            break
    for ymax in range(img.shape[1]-1, 0, -1):
        if np.sum(img[:,ymax,0]) > 0.01:
            break
    no_pad = img[xmin:xmax,ymin:ymax+1,...]
    return no_pad

def remove_pad_save_to_new_dir(input_dir, output_dir, file_format='.png', num_images=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in os.listdir(input_dir)[:num_images]:
        if file != '.ipynb_checkpoints':
            # Load Image
            filepath = os.path.join(input_dir, file)
            if filepath.endswith('.npy'):
                array = np.load(filepath)
            else:
                array = imageio.imread(filepath)
                array = np.array(array)
            new_array = remove_padding(array)
            new_array = exposure.rescale_intensity(new_array, in_range='image', out_range=(0.0,255.0)).astype('uint8')
            new_filepath = Path(os.path.join(output_dir, file))
            new_filepath = Path(os.path.abspath(new_filepath.with_suffix('')) + file_format)
            if file_format == '.npy':
                np.save(new_filepath, new_array, allow_pickle=True, fix_imports=True)
            else:
                imageio.imwrite(new_filepath, new_array)