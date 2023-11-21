"""
Functions for work with masks/labels and multidimensional arrays using binary masks and label

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
from scipy import optimize


def pb_exp_correction(input_img:np.ndarray, mask:np.ndarray, method:str='exp'):
    """ Image series photobleaching correction by exponential fit.
    
    Correction proceeds by masked area of interest, not the whole frame to prevent autofluorescence influence.

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        input image series
    mask: ndarray [x,y]
        mask of region of interest, must be same size with image frames
    method: str
        method for correction, exponential (`exp`) or bi-exponential (`bi_exp`)

    Returns
    -------
    corrected_img: ndarray [t,x,y]
        corrected image series
    bleach_coefs: ndarray [t]
        array of correction coeficients for each frame
    r_val: float
        R-squared value of exponential fit

    """
    exp = lambda x,a,b: a * np.exp(-b * x)
    bi_exp = lambda x,a,b,c,d: (a * np.exp(-b * x)) + (c * np.exp(-d * x))

    if method == 'exp':
        func = exp
    elif method == 'bi_exp':
        func = bi_exp
    else:
        raise ValueError('Incorrect method!')

    bleach_profile = np.mean(input_img, axis=(1,2), where=mask)
    x_profile = np.linspace(0, bleach_profile.shape[0], bleach_profile.shape[0])

    popt,_ = optimize.curve_fit(func, x_profile, bleach_profile)
    bleach_fit = np.vectorize(func)(x_profile, *popt)
    bleach_coefs =  bleach_fit / bleach_fit.max()
    bleach_coefs_arr = bleach_coefs.reshape(-1, 1, 1)
    corrected_image = input_img/bleach_coefs_arr

    _,_,r_val,_,_ = stats.linregress(bleach_profile, bleach_fit)

    return corrected_image, bleach_coefs, r_val


def proc_mask(input_img:np.ndarray,
              proc_sigma:float=1.0, win_size:int=801, k_val:float=1e-4, r_val:float=0.5,
              soma_mask:bool=False, soma_th:float=.5, soma_ext:int=100,
              ext_fin_mask:bool=False, proc_ext:int=5,
              select_largest_mask:bool=False):
    """
    __NB: used by default in most registration types!__

    Mask for single neuron images with bright soma region
    with Sauvola local threshold. Func drops soma and returns the mask for processes only.

    In the original method, a threshold T is calculated
    for every pixel in the image using the following formula
    (`m(x,y)` is mean and `s(x,y)` is SD in rectangular window):

    `
    T = m(x,y) * (1 + k * ((s(x,y) / r) - 1))
    `

    Parameters
    ----------
    input_img: ndarray
        input img for mask creation,
        recomend choose max int projection of brighter channel (GFP/YFP in our case)
    proc_sigma: float
        sigma value for gaussian blur of input image
    win_size: int
        Sauvola local threshold parameter, window size specified as a single __odd__ integer
    k_val: float
        Sauvola local threshold parameter, value of the positive parameter `k`
    r_val: float  
        Sauvola local threshold parameter, value of `r`, the dynamic range of standard deviation
    soma_mask: boolean, optional
        if True brighter region on image will identified and masked as soma
    soma_th: float, optional
        soma detection threshold in % of max img int
    soma_ext: int, optional
        soma mask extension size in px
    ext_fin_mask: boolean, optional
        if `True` - final processes mask will be extended on proc_ext value
    proc_ext: int
        neuron processes mask extention value in px
    select_largest_mask: bool, optional
        if `True` - final mask will contain the largest detected element only

    Returns
    -------
    proc_mask: ndarray, dtype boolean
        neuron processes mask
    
    """
    input_img = filters.gaussian(input_img, sigma=proc_sigma)

    # processes masking    
    proc_th = filters.threshold_sauvola(input_img, window_size=win_size, k=k_val, r=r_val)
    proc_mask = input_img > proc_th

    # mask extention
    if ext_fin_mask:
        proc_dist = ndi.distance_transform_edt(~proc_mask, return_indices=False)
        proc_mask = proc_dist <= proc_ext

    # soma masking
    if soma_mask:
        soma_region = np.copy(input_img)
        soma_region = soma_region > soma_region.max() * soma_th
        soma_region = morphology.opening(soma_region, footprint=morphology.disk(5))
        soma_dist = ndi.distance_transform_edt(~soma_region, return_indices=False)
        soma_region = soma_dist < soma_ext
        proc_mask[soma_region] = 0

    # minor mask elements filtering
    if select_largest_mask:
        def largest_only(raw_mask):
            # get larger mask element
            element_label = measure.label(raw_mask)
            element_area = {element.area : element.label for element in measure.regionprops(element_label)}
            larger_mask = element_label == element_area[max(element_area.keys())]
            return larger_mask
        proc_mask = largest_only(proc_mask)

    return proc_mask


def proc_mask_otsu(input_img:np.ndarray,
              soma_mask:bool=True, soma_th:float=.5, soma_ext:int=20,
              proc_ext:int=5, ext_fin_mask:bool=True, proc_sigma:float=.5):
    """ Mask for single neuron images with bright soma region with simple Otsu thresholding.
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


def mask_connection(input_master_mask:np.ndarray, input_minor_mask:np.ndarray):
    """ Function to filter two masks by overlay.

    Could be used in stand-alone mode as a static method of the WTvsMut class.

    Parameters
    ----------
    input_master_mask: ndarray [x,y]
        boolean mask for filtering with a greater number/area of insertions,
        typically the mask of wild-type NCS insertions
    input_minor_mask: ndarray [x,y]
        boolean mask with a lower number/area of insertions,
        typically the mask of mutant NCS insertions

    Returns
    -------
    fin_mask: ndarray [x,y]
        boolean mask that includes only the elements from 'input_wt_mask'
        that overlap with the elements from 'input_mut_mask'  
    fin_label: ndarray [x,y]
        label image for `fin_mask`

    """ 
    master_label, master_num = ndi.label(input_master_mask)

    sums = ndi.sum(input_minor_mask, master_label, np.arange(master_num+1))
    connected = sums > 0
    debris_mask = connected[master_label]

    fin_mask = np.copy(input_master_mask)
    fin_mask[~debris_mask] = 0

    fin_label, fin_num = ndi.label(fin_mask)

    return fin_mask, fin_label


def mask_along_frames(series: np.ndarray, mask: np.ndarray):
    masked_series = []

    for frame in series:
        back_mean = np.mean(frame, where=mask)
        masked_frame = np.copy(frame)
        masked_frame[~mask] = back_mean / 4
        masked_series.append(masked_frame)

    return np.asarray(masked_series)


def background_extraction_along_frames(series: np.ndarray, mask: np.ndarray):
    masked_series = []

    for frame in series:
        back_mean = np.mean(frame, where=mask)
        masked_frame = np.copy(frame)
        masked_frame = masked_frame - back_mean
        masked_series.append(masked_frame)

    return np.asarray(masked_series)


def label_prof_arr(input_label: np.ndarray, input_img_series: np.ndarray,
                   f0_win:int=3):
    """ Calc labeled ROIs profiles for time series. 

    Parameters
    ----------
    input_label: ndarray [x,y]
        label image
    input_img_series: ndarray [t,x,y]
        image time series, frames must be the same size with label array
    f0_win: int, optional
        number of points from profile start to calc F0

    Returns
    -------
    prof_df_arr: ndarray [dF_value,t]
        ΔF/F profiles for each label elements
    prof_arr: ndarray [intensity_value,t]
        intensity profiles for each label elements
 
    """
    output_dict = {}
    prof_arr = []
    prof_df_arr=[]
    for label_num in np.unique(input_label)[1:]:
        region_mask = input_label == label_num
        prof = np.mean(input_img_series, axis=(1,2), where=region_mask)
        F_0 = np.mean(prof[:f0_win])
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


def series_derivate(input_img:np.ndarray, mask:np.ndarray,
                    left_win=1, space:int=0, right_win:int=1):
    """ Calculation of intensity derivative profile for image series.
    
    Parameters
    ----------
    input_img: ndarray [t,x,y]
        input image series
    mask: ndarray, dtype boolean
        cell mask
    left_win: int
        number of frames for left image
    space: int
        number of skipped frames between left and right images
    left_win: int
        number of frames for left image

    Returns
    -------
    der_arr_win: ndarray [i]
        1D array of derivate intensity calculated
        for framed images (left-skip-right), 0-1 normalized
    prof_df_arr: ndarray [i]
        1D array of derivative intensity calculated frame by frame
        and smoothed with moving average (n=3), 0-1 normalized

    """
    der_img = []
    der_arr_win = []
    der_arr_point = []
    for i in range(input_img.shape[0] - (left_win+space+right_win)):
        der_frame = np.mean(input_img[i+left_win+space:i+left_win+space+right_win+1], axis=0) - np.mean(input_img[i:i+left_win+1], axis=0) 
        der_val = np.sum(np.abs(der_frame), where=mask)
        
        der_frame_point = input_img[i+1] - input_img[i]
        der_val_point = np.sum(np.abs(der_frame_point), where=mask)

        der_img.append(der_frame)
        der_arr_win.append(der_val)
        der_arr_point.append(der_val_point)

    der_img = np.asarray(der_img)

    der_arr_win = np.asarray(der_arr_win)
    der_arr_win = (der_arr_win - np.min(der_arr_win)) / (np.max(der_arr_win)-np.min(der_arr_win))
    der_arr_win = np.pad(der_arr_win, (left_win+space+right_win), constant_values=0)[:input_img.shape[0]]

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    der_arr_point = np.asarray(der_arr_point)
    der_arr_point = (der_arr_point - np.min(der_arr_point)) / (np.max(der_arr_point)-np.min(der_arr_point))
    der_arr_point = moving_average(der_arr_point)
    der_arr_point = np.pad(der_arr_point, 5, constant_values=0)[:input_img.shape[0]]

    return der_arr_win, der_arr_point


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