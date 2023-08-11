"""
E app calculation for FRET estimation

Based on Zal and Gascoigne, 2004, doi: 10.1529/biophysj.103.022087

"""

import os
import sys
import glob
import yaml

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


def Fc_calc(dd_img, da_img, aa_img, a, b, c, d):
    Fc_img = []
    for frame_num in range(dd_img.shape[0]):
        DD_frame = dd_img[frame_num]
        DA_frame = da_img[frame_num]
        AA_frame = aa_img[frame_num]

        Fc_frame = DA_frame - a*(AA_frame - c*DD_frame) - d*(DD_frame - b*AA_frame)
        Fc_img.append(Fc_frame)

    return np.asarray(Fc_img)


def E_app_calc(fc_img, dd_img, G, mask):
    R_img = []
    E_app_img = []
    for frame_num in range(fc_img.shape[0]):
        Fc_frame = fc_img[frame_num]
        DD_frame = dd_img[frame_num]

        R_frame = Fc_frame / DD_frame
        R_frame = R_frame
        E_app_frame = R_frame / (R_frame + G)

        E_app_frame[E_app_frame < 0] = 0
        E_app_frame[~mask] = 0

        R_img.append(R_frame)
        E_app_img.append(E_app_frame)

    return np.asarray(R_img), np.asarray(E_app_img)


def E_cor_calc():
    pass