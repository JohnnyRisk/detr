import os
import cv2
from tqdm import tqdm
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops

from skimage.morphology import label


def convert_segmentation_to_list(contour):
    coord_list = []
    coords = contour.squeeze(axis=1)
    for i in range(coords.shape[0]):
        coord_list.append(float(coords[i][0]))
        coord_list.append(float(coords[i][1]))
    return [coord_list]


### NEED TO CHECK THIS TOMORROW FOR FORMATTING
def convert_bbox_to_XYHW(contour):
    min_x = np.min(contour.squeeze().T[0])
    min_y = np.min(contour.squeeze().T[1])
    max_x = np.max(contour.squeeze().T[0])
    max_y = np.max(contour.squeeze().T[1])
    w = max_x - min_x
    h = max_y - min_y
    return [float(min_x), float(min_y), float(w), float(h)]


def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(500, 500)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((500, 500), dtype=np.int16)
    # if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
