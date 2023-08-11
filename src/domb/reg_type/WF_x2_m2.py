"""
Class for wide-field registrations with 2 excitation wl and 2 emission ch.Class for wide-field registrations
with 2 excitation wavelengths and 2 emission pass-band recordings

Optimized for individual neurons imaging

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

from ..util import masking
from ..util import plot


class WF_2x_2m():
    def __init__(self, img_path, img_name, sigma=.5, **kwargs):
        self.img_raw = io.imread(img_path)
        self.frame_max = filters.gaussian(np.max(self.img_raw, axis=(0,3)), sigma=.5)
        self.img_name = img_name

        # self.a = abcd_list[0]
        # self.b = abcd_list[1]
        # self.c = abcd_list[2]
        # self.d = abcd_list[3]

        # chennels split
        self.ch0_img = np.asarray([filters.gaussian(frame, sigma=sigma) \
                                   for frame in self.img_raw[:,:,:,0]])  # 435-CFP  DD
        self.ch1_img = np.asarray([filters.gaussian(frame, sigma=sigma) \
                                   for frame in self.img_raw[:,:,:,1]])  # 435-YFP  DA
        self.ch2_img = np.asarray([filters.gaussian(frame, sigma=sigma) \
                                   for frame in self.img_raw[:,:,:,2]])  # 505-CFP  AD
        self.ch3_img = np.asarray([filters.gaussian(frame, sigma=sigma) \
                                   for frame in self.img_raw[:,:,:,3]])  # 505-YFP  AA

        self.cfp_img = self.ch0_img
        self.yfp_img = self.ch3_img

        # cell masking
        self.proc_mask = masking.proc_mask(self.frame_max, proc_ext=5, **kwargs)
        self.narrow_proc_mask = masking.proc_mask(self.frame_max, proc_ext=0, **kwargs)


    @staticmethod
    def __linear_unmixing():
        pass