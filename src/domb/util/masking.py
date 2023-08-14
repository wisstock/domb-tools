"""
Functions for masking multidimensional arrays using binary masks and labels

"""

import numpy as np
from numpy import ma
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import plotly.express as px

from skimage.util import montage
from skimage.filters import rank
from skimage import morphology
from skimage import exposure
from skimage import measure
from skimage import filters
from skimage import io
from skimage import transform
from skimage import registration

from scipy import ndimage
from scipy import signal
from scipy import stats
from scipy import ndimage as ndi



def proc_mask(input_img, soma_mask=True, soma_th=0.5, soma_ext=20, proc_ext=5):
    """
    Mask for single neuron images with bright soma region.
    Fiunc drops soma and returns the mask for processes only.
    
    """
    # soma masking
    if soma_mask:
        soma_region = np.copy(input_img)
        soma_region = soma_region > soma_region.max() * soma_th
        soma_region = morphology.opening(soma_region, footprint=morphology.disk(5))
        # soma_region = morphology.dilation(soma_region, footprint=morphology.disk(10))
        soma_dist = ndimage.distance_transform_edt(~soma_region, return_indices=False)
        soma_mask = soma_dist < soma_ext
        input_img = ma.masked_where(soma_mask, input_img)

        th = filters.threshold_otsu(input_img.compressed())
    else:
        th = filters.threshold_otsu(input_img)

    # processes masking
    proc_mask = input_img > th
    proc_mask = morphology.closing(proc_mask, footprint=morphology.disk(5))
    proc_dist = ndimage.distance_transform_edt(~proc_mask, return_indices=False)
    proc_mask_fin = proc_dist <= proc_ext
    proc_mask_fin[soma_mask] = 0

    return proc_mask_fin


def mask_along_frames(series, mask):
    """
    Time series masking along the time axis.
    Func fills the area around the mask with a fixed value (1/4 of outer mean intensity).

    """
    masked_series = []

    for frame in series:
        back_mean = np.mean(frame, where=mask)
        masked_frame = np.copy(frame)
        masked_frame[~mask] = back_mean / 4
        masked_series.append(masked_frame)

    return np.asarray(masked_series)


def pb_corr(img, base_win=4, mode='exp'):
    """ ftame series photobleaxhing crrection with exponential fit

    """
    if mode == 'exp':
        x = np.arange(0, img.shape[0], 1, dtype='float')
        y = np.asarray([np.mean(i) for i in img])

        p = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
        a = np.exp(p[1])
        b = p[0]

        fit = a * np.exp(b * x)
        fit_arr = np.asarray([z+f for z, f in zip(np.zeros_like(img), fit)])
        img_corr = (img*fit_arr) + np.mean(img[:base_win], axis=0)

        print(a, b)

        plt.plot([np.mean(i) for i in img_corr], label='corr')
        plt.plot([np.mean(i) for i in img], linestyle='--', label='raw')
        plt.plot(fit, linestyle=':', label='fit')
        plt.legend()
        plt.show()
        
    return img_corr