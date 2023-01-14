import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread, imshow, concatenate_images
from skimage.color import rgb2gray
from PIL import Image as Img
from skimage import color

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten

def plot_image_mask_path(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.resize(img_gray, (256,256), interpolation=cv2.INTER_AREA)

    mask = cv2.imread(mask_path)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray = cv2.resize(mask_gray, (256,256), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(1,3, figsize=(20,20))

    axes[0].set_title("mask", color = 'red')
    axes[0].set_ylabel(mask_path)
    axes[0].imshow(mask)

    axes[1].set_title("Original Image", color='red')
    axes[1].imshow(img, cmap='gray')

    result_image = color.label2rgb(mask_gray, np.array(img_gray)/np.array(img_gray).max())
    axes[2].set_title('Mask over Image', color='red')
    axes[2].imshow(result_image)

def dice_coefficients(y_true, y_pred, smooth = 100):
  y_true_flattened = Flatten()(y_true)
  y_pred_flattened = Flatten()(y_pred)
  intersection = K.sum(y_true_flattened * y_pred_flattened)
  union = K.sum(y_true_flattened) + K.sum(y_pred_flattened)
  return (2 * intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=100):
  return -dice_coefficients(y_true, y_pred, smooth=100)

def iou(y_true, y_pred, smooth = 100):
  intersection = K.sum(y_true * y_pred)
  sum = K.sum(y_true + y_pred)
  iou = (intersection + smooth) / (sum - intersection + smooth)
  return iou

def jaccard_distance(y_true, y_pred):
  y_true_flatten = Flatten()(y_true)
  y_pred_flatten = Flatten()(y_pred)
  return -iou(y_true_flatten, y_pred_flatten)







