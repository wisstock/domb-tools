"""
Functions for masking multidimensional arrays using binary masks and label

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



def proc_mask(input_img:np.ndarray,
              soma_mask:bool=True, soma_th:float=.5, soma_ext:int=20,
              proc_ext:int=5, ext_fin_mask:bool=True, proc_sigma:float=.5):
    """ Mask for single neuron images with bright soma region.
    Fiunc drops soma and returns the mask for processes only.

    Parameters
    ----------
    input_img: ndarray
        input img for mask creation,
        recomend choose max int projection of brighter channel (YFP)
    soma_mask: boolean, optional
        if True brighter region on image will identified and masked as soma
    soma_th: float, optional
        soma detection threshold in % of max img int
    soma_ext: int, optional
        soma mask extension size in px
    proc_ext: int
        cell processes mask extention in px
    ext_fin_mask: bool, optional
        if True - final processes mask will be extended on proc_ext val
    proc_sigma: float
        sigma value for gaussian blur of input image

    Returns
    -------
    proc_mask_fin: ndarray, dtype boolean
       extented cell processes mask
    
    """
    # soma masking
    input_img = filters.gaussian(input_img, sigma=proc_sigma)
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
    if ext_fin_mask:
        proc_dist = ndimage.distance_transform_edt(~proc_mask, return_indices=False)
        proc_mask_fin = proc_dist <= proc_ext
    else:
        proc_mask_fin = proc_mask
    proc_mask_fin[soma_mask] = 0

    return proc_mask_fin


def mask_along_frames(series: np.ndarray, mask: np.ndarray):
    """ Time series masking along the time axis.
    Func fills the area around the mask with a fixed value (1/4 of outer mean intensity).

    """
    masked_series = []

    for frame in series:
        back_mean = np.mean(frame, where=mask)
        masked_frame = np.copy(frame)
        masked_frame[~mask] = back_mean / 4
        masked_series.append(masked_frame)

    return np.asarray(masked_series)


def label_prof_arr(input_label: np.ndarray, input_img_series: np.ndarray):
    """ Calc labeled ROIs profiles for time series 

    Parameters
    ----------
    input_label: ndarray [x,y]
        label image
    input_img_series: ndarray [t,x,y]
        image time series, frames must be the same size with label array

    Returns
    -------
    prof_arr: ndarray [intensity_value,t]
        intensity profiles for each label elements
    prof_df_arr: ndarray [dF_value,t]
        ΔF/F profiles for each label elements
 
    """
    output_dict = {}
    prof_arr = []
    prof_df_arr=[]
    for label_num in range(1, np.max(input_label)+1):
        region_mask = input_label == label_num
        prof = np.asarray([np.mean(ma.masked_where(~region_mask, img)) for img in input_img_series])
        F_0 = np.mean(prof[:3])
        df_prof = (prof-F_0)/F_0
        output_dict.update({label_num:[prof, df_prof]})

        prof_arr.append(prof)
        prof_df_arr.append(df_prof)
        
    
    return np.asarray(prof_df_arr), np.asarray(prof_arr)


def trans_prof_arr(input_total_mask: np.ndarray,
                   input_label: np.ndarray,
                   input_img_series: np.ndarray):
    """ Calc ratio mask int/ROIs int for time series 

    Parameters
    ----------
    input_total_mask: ndarray, dtype boolean
        cell mask
    input_label: ndarray
        ROIs label array
    input_img_series: ndarray
        Series of images as 3D array with dim. order (time,x,y),
        each frame should be same size wiht cell mask array.

    Returns
    -------
    trans_rois_arr: ndarray
        2D array with dim. order (t,translocation profile)
    prof_df_arr: ndarray
        array with ratio "all ROIs int/cell mask int" for each frame
 
    """
    trans_rois_arr = []
    for label_num in range(1, np.max(input_label)+1):
        region_mask = input_label == label_num
        prof = np.asarray([np.sum(img, where=region_mask) / np.sum(img, where=input_total_mask) \
                           for img in input_img_series])
        trans_rois_arr.append(prof)

    total_rois = input_total_mask !=0    
    trans_total_arr = [np.sum(img, where=total_rois)/np.sum(img, where=input_total_mask) \
                       for img in input_img_series]

    return np.asarray(trans_rois_arr), np.asarray(trans_total_arr)


def label_prof_dist(input_label: np.ndarray,
                    input_img_series: np.ndarray,
                    input_dist_img: np.ndarray):
    output_dict = {}
    for label_num in range(1, np.max(input_label)+1):
        region_mask = input_label == label_num
        prof = np.asarray([np.mean(ma.masked_where(~region_mask, img)) for img in input_img_series])
        F_0 = np.mean(prof[:5])
        df_prof = (prof-F_0)/F_0
        dist = round(np.mean(ma.masked_where(~region_mask, input_dist_img)), 1)
        output_dict.update({dist:[label_num, prof, df_prof]})
    
    return output_dict


def sorted_prof_arr_calc(input_prof_dict: np.ndarray,
                         min_dF=0.25, mid_filter=True, mid_filter_div=2):
    sorted_dist = list(input_prof_dict.keys())
    sorted_dist.sort(key=float)

    prof_arr = []
    prof_id = []
    prof_dist = []
    for dist in sorted_dist:
        roi_data = input_prof_dict[dist]
        roi_id = roi_data[0]
        # roi_prof = roi_data[1]
        roi_prof_df = roi_data[2]

        if mid_filter:
            max_idx = np.argmax(roi_prof_df)
            if max_idx > len(roi_prof_df) // mid_filter_div:
                continue
            
        if np.max(roi_prof_df) > min_dF:
            prof_id.append(roi_id)
            prof_arr.append(roi_prof_df)
            prof_dist.append(dist)

    prof_id = np.asarray(prof_id)
    prof_arr = np.asarray(prof_arr)
    prof_dist = np.asarray(prof_dist)

    return prof_arr


def pb_corr(img: np.ndarray, base_win=4, mode='exp'):
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