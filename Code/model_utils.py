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
from pathlib import Path
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Add, Input
from tensorflow.keras.layers import BatchNormalization, UpSampling2D, Concatenate, Conv2DTranspose
from skimage import exposure
import model_utils
##################################################################################################################################
'''
MODEL UTILS:
'''
##################################################################################################################################
# Custom Metrics:

def normalize(tensor):
    # Normalizes Tensor from 0-1
    return tf.math.divide_no_nan(tf.math.subtract(tensor, tf.math.reduce_min(tensor)), 
                                 tf.math.subtract(tf.math.reduce_max(tensor), tf.math.reduce_min(tensor)))
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(model_utils.normalize, y_true)
    y_pred_norm = tf.map_fn(model_utils.normalize, y_pred)
    PSNR = tf.image.psnr(y_true_norm, y_pred_norm, max_pixel)
    return PSNR

def SSIM(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(model_utils.normalize, y_true)
    y_pred_norm = tf.map_fn(model_utils.normalize, y_pred)
    SSIM = tf.image.ssim(y_true_norm,y_pred_norm,max_pixel,filter_size=11,
                         filter_sigma=1.5,k1=0.01,k2=0.03)
    return SSIM
def KLDivergence(y_true, y_pred):
    return tf.losses.KLDivergence()(y_true, y_pred)

def TV(y_true, y_pred):
    img_list = [y_true[0,...], y_pred[0,...]]
    images = tf.stack(img_list)
    loss = tf.math.reduce_sum(tf.image.total_variation(images))
    return loss
def SavingMetric(y_true, y_pred):
    # Combines Insight from SSIM and PSNR
    SSIM = model_utils.SSIM(y_true, y_pred)
    PSNR = model_utils.PSNR(y_true, y_pred)
    # Normalize for Minimization:
    SSIM_norm = 1 - SSIM
    PSNR_norm = (40 - PSNR)/275
    loss = SSIM_norm + PSNR_norm
    return loss

# Model Loss Function:
def model_loss(B1=1.0, B2=0.01, B3=0.0, B4=0.0):
    @tf.function
    def loss_func(y_true, y_pred):
        F_mag_true = tf.map_fn(FFT_mag, y_true)
        F_mag_pred = tf.map_fn(FFT_mag, y_pred)
        #F_phase_true = tf.map_fn(FFT_phase, y_true)
        #F_phase_pred = tf.map_fn(FFT_phase, y_pred)
        if tf.executing_eagerly():
            MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred).numpy()
            # Fourier Loss
            F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred).numpy()
            #F_phase_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_phase_true, F_phase_pred).numpy()
            # Max Absolute Difference
            MaxAbsDiff_Loss = tf.math.reduce_max(tf.math.abs(tf.math.subtract(y_true, y_pred))).numpy()
        else:
            MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
            # Fourier Loss
            F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred)
            #F_phase_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_phase_true, F_phase_pred)
            # Max Absolute Difference
            MaxAbsDiff_Loss = tf.math.reduce_max(tf.math.abs(tf.math.subtract(y_true, y_pred)))
        F_mag_MAE_Loss = tf.cast(F_mag_MAE_Loss, dtype=tf.float32)
        #F_phase_MAE_Loss = tf.cast(F_phase_MAE_Loss, dtype=tf.float32)
        loss = B1*MAE_Loss + B2*F_mag_MAE_Loss + B4*MaxAbsDiff_Loss#+ B3*F_phase_MAE_Loss
        return loss
    return loss_func
def FFT_mag(input):
    # FFT Function to be performed for each instance in batch
    real = input
    imag = tf.zeros_like(input)
    out = tf.abs(tf.signal.fft2d(tf.complex(real, imag)[:, :, 0]))
    return out
# def FFT_phase(input):
#     # FFT Function to be performed for each instance in batch
#     real = input
#     imag = tf.zeros_like(input)
#     out = tf.math.angle(tf.signal.fft2d(tf.complex(real, imag)[:, :, 0]))
#     return out

# Model Loss Function:
def model_loss_experimental(B1=1.0, B2=0.0, B3=0.0):
    @tf.function
    def loss_func(y_true, y_pred):
        F_mag_true = tf.map_fn(FFT_mag, y_true)
        F_mag_pred = tf.map_fn(FFT_mag, y_pred)
        if tf.executing_eagerly():
            MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred).numpy()
            # Fourier Loss
            F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred).numpy()
            # Max Absolute Difference
            MaxAbsDiff_Loss = tf.math.reduce_max(tf.math.abs(tf.math.subtract(y_true, y_pred))).numpy()
            # SSIM and PSNR
            saving_metric = model_utils.SavingMetric(y_true, y_pred).numpy()
        else:
            MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
            # Fourier Loss
            F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred)
            # Max Absolute Difference
            MaxAbsDiff_Loss = tf.math.reduce_max(tf.math.abs(tf.math.subtract(y_true, y_pred)))
            # SSIM and PSNR
            saving_metric = model_utils.SavingMetric(y_true, y_pred)
        F_mag_MAE_Loss = tf.cast(F_mag_MAE_Loss, dtype=tf.float32)
        #F_phase_MAE_Loss = tf.cast(F_phase_MAE_Loss, dtype=tf.float32)
        loss = B1*MAE_Loss + B2*F_mag_MAE_Loss + B3*saving_metric
        return loss
    return loss_func