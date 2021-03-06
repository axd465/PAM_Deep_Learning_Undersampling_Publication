# Customary Imports:
import tensorflow as tf
assert '2.' in tf.__version__  # make sure you're using tf 2.0
import numpy as np
import matplotlib.pyplot as plt
import math
import string
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
import copy
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

# Added Import Statements:
import patchwork_alg
from patchwork_alg import expand_image
from patchwork_alg import expand_image_with_interp
from patchwork_alg import fix_boundaries
from patchwork_alg import apply_model_patchwork
from patchwork_alg import remove_peak
from model_utils import PSNR
from model_utils import SSIM
from model_utils import KLDivergence
import test_statistics
##################################################################################################################################
'''
COMPUTING TEST STATISTICS FOR INTERPOLATION:
'''
def obtain_test_stats(model, input_dir, downsampling_ratio = (1,5), shape_for_model = (128,128), 
                      buffer = 20, contrast_enhance = (0.05, 99.95), interp=False, gauss_blur_std=None,
                      output_dir = None, file_format = '.tif', remove_pad = False):
    '''
    This function loops through a directory and computes various test statistics (comparing DL and Bicubic
    Interpolation) for each image in the directory. These test statistics are then exported as a dictionary.
    '''
    file_list = os.listdir(input_dir)
    metrics = {'FILENAME':[], 'PSNR':[], 'SSIM':[], 'MEAN ABSOLUTE ERROR':[], 'MEAN SQUARED ERROR':[], 
               'KL DIVERGENCE':[], 'MEAN SQUARED LOG ERROR':[], 'LOG COSH ERROR':[], 
               'POISSON LOSS':[]}
    stats = {'Deep Learning':copy.deepcopy(metrics), 'Bicubic Interpolation':copy.deepcopy(metrics)}
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            dl_dir = os.path.join(output_dir, 'deep_learning')
            os.mkdir(dl_dir)
            bc_dir = os.path.join(output_dir, 'bicubic_interp')
            os.mkdir(bc_dir)
            full_sample_dir = os.path.join(output_dir, 'fully_sampled')
            os.mkdir(full_sample_dir)
            latent_dir = os.path.join(output_dir, 'latent_imgs')
            os.mkdir(latent_dir)
        else:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
    file_count = 1
    for file in file_list:
        if file != '.ipynb_checkpoints':
            # Load Image
            filename = os.fsdecode(file)
            filepath = os.path.join(input_dir, filename)
            if filepath.endswith('.npy'):
                img = np.load(filepath)
            else:
                img = imageio.imread(filepath)
                img = np.array(img, dtype=np.float32)

            # COMPUTING STATISTICS:
            full_samp_img = exposure.rescale_intensity(img[...,0], in_range='image', out_range=(0.0,1.0))
            '''
            # If using color images
            if len(img.shape)>2:
                img = np.mean(img, axis = 2)
            '''
            # Resizing Images To Balance Downsampling Ratio Along Axes (one axis already downsampled):
            i_ratio = 2
            j_ratio = 1
            full_samp_img_shape = (full_samp_img.shape[0]*i_ratio, full_samp_img.shape[1]*j_ratio)
            full_samp_img = skimage.transform.resize(full_samp_img, output_shape=full_samp_img_shape, order=3, 
                                                     mode='reflect', cval=0, clip=True, preserve_range=True, 
                                                     anti_aliasing=True, anti_aliasing_sigma=None)
            # Recover Latent Image:
            latent_image = np.zeros((int(np.ceil(full_samp_img.shape[0]/downsampling_ratio[0])), 
                                     int(np.ceil(full_samp_img.shape[1]/downsampling_ratio[1]))))
            i_count = 0
            for i in range(0, full_samp_img.shape[0], downsampling_ratio[0]):
                j_count = 0
                for j in range(0, full_samp_img.shape[1], downsampling_ratio[1]):
                    latent_image[i_count, j_count] = full_samp_img[i, j]
                    if j%downsampling_ratio[1]==0:
                        j_count += 1
                if i%downsampling_ratio[0]==0:
                    i_count += 1
            deep_image = apply_model_patchwork(model, down_image = latent_image, downsampling_ratio = downsampling_ratio, 
                                               downsampling_axis = 'both', shape_for_model = shape_for_model, 
                                               buffer = buffer, output_shape = full_samp_img.shape, interp = interp, 
                                               remove_pad = remove_pad, gauss_blur_std = gauss_blur_std)
            #'''
            deep_image = skimage.img_as_float(deep_image)
            p1, p2 = np.percentile(deep_image, contrast_enhance)
            deep_image = exposure.rescale_intensity(deep_image, in_range=(p1, p2), out_range=(0.0,1.0))
            #'''

            # COMPARISON TO INTERPOLATION:
            # From https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
            # and https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
            interp_img = skimage.transform.resize(latent_image, output_shape=full_samp_img.shape, order=3, mode='reflect', 
                                                  cval=0, clip=True, preserve_range=True, anti_aliasing=True, anti_aliasing_sigma=None)
            '''
            # To Show Images
            figsize = (20,20)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.imshow(deep_image, cmap = 'gray')
            plt.title('DL Image')
            plt.show()
            '''
            # Save Images to Output Directory
            if output_dir is not None:
                test_statistics.save_img(deep_image, file, dl_dir, file_format)
                test_statistics.save_img(interp_img, file, bc_dir, file_format)
                test_statistics.save_img(full_samp_img, file, full_sample_dir, file_format)
                test_statistics.save_img(latent_image, file, latent_dir, file_format)

            # Quantitative Measurements
            deep_image = deep_image[..., None]
            deep_image = tf.image.convert_image_dtype(deep_image[None, ...], tf.float32)
            interp_img = interp_img[..., None]
            interp_img = tf.image.convert_image_dtype(interp_img[None, ...], tf.float32)
            full_samp_img = full_samp_img[..., None]
            full_samp_img = tf.image.convert_image_dtype(full_samp_img[None, ...], tf.float32)
            '''
            ---------------------------------------
            ##########LIST OF STATISTICS:##########
            ---------------------------------------
            '''
            # FILENAME:
            stats['Deep Learning']['FILENAME'].extend([file])
            stats['Bicubic Interpolation']['FILENAME'].extend([file])
            # PSNR:
            stats['Deep Learning']['PSNR'].extend(PSNR(full_samp_img, deep_image).numpy())
            stats['Bicubic Interpolation']['PSNR'].extend(PSNR(full_samp_img, interp_img).numpy())
            # SSIM:
            stats['Deep Learning']['SSIM'].extend(SSIM(full_samp_img, deep_image).numpy())
            stats['Bicubic Interpolation']['SSIM'].extend(SSIM(full_samp_img, interp_img).numpy())
            # MAE:
            MAE = tf.keras.losses.MeanAbsoluteError()
            stats['Deep Learning']['MEAN ABSOLUTE ERROR'].extend([MAE(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['MEAN ABSOLUTE ERROR'].extend([MAE(full_samp_img, interp_img).numpy()])
            # MSE:
            MSE = tf.keras.losses.MeanSquaredError()
            stats['Deep Learning']['MEAN SQUARED ERROR'].extend([MSE(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['MEAN SQUARED ERROR'].extend([MSE(full_samp_img, interp_img).numpy()])
            # KL DIVERGENCE:
            stats['Deep Learning']['KL DIVERGENCE'].extend([KLDivergence(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['KL DIVERGENCE'].extend([KLDivergence(full_samp_img, interp_img).numpy()])
            # MEAN SQUARED LOGARITHMIC ERROR:
            MSLE = tf.keras.losses.MeanSquaredLogarithmicError()
            stats['Deep Learning']['MEAN SQUARED LOG ERROR'].extend([MSLE(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['MEAN SQUARED LOG ERROR'].extend([MSLE(full_samp_img, interp_img).numpy()])
            # LOG COSH ERROR:
            LCOSH = tf.keras.losses.LogCosh()
            stats['Deep Learning']['LOG COSH ERROR'].extend([LCOSH(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['LOG COSH ERROR'].extend([LCOSH(full_samp_img, interp_img).numpy()])
            # POISSON LOSS:
            PL = tf.keras.losses.Poisson()
            stats['Deep Learning']['POISSON LOSS'].extend([PL(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['POISSON LOSS'].extend([PL(full_samp_img, interp_img).numpy()])

            print(f'Done with file {file_count} out of {len(file_list)}')
            file_count += 1
    return stats

def save_img(img, filename, output_dir, file_format='.tif'):
        filepath = os.path.join(output_dir, filename)
        # Save Image
        if file_format == '.npy':
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(file_format)
            np.save(new_filepath, img, allow_pickle=True, fix_imports=True)
        elif file_format == '.tif':
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(file_format)
            imageio.imwrite(new_filepath, img)
        else:
            temp = exposure.rescale_intensity(img, in_range='image', 
                                              out_range=(0.0,255.0)).astype('uint8')
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(file_format)
            imageio.imwrite(new_filepath, temp)