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
        self.frame_max = filters.gaussian(np.mean(self.img_raw[:,:,:,3], axis=0),
                                          sigma=1)
        self.img_name = img_name

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

        self.cfp_mean_img_raw = np.mean(self.img_raw[:,:,:,0], axis=0)
        self.yfp_mean_img_raw = np.mean(self.img_raw[:,:,:,3], axis=0)

        # mask creation
        self.proc_mask = masking.proc_mask(self.frame_max, proc_ext=5, **kwargs)
        self.narrow_proc_mask = masking.proc_mask(self.frame_max, proc_ext=0, **kwargs)

        # img maskings
        self.mask_arr = np.asarray([z+self.proc_mask for z in np.zeros_like(self.cfp_img)], dtype='bool')
        self.masked_cfp_img = np.copy(self.cfp_img)
        self.masked_cfp_img[~self.mask_arr] = 0

        self.masked_yfp_img = np.copy(self.yfp_img)
        self.masked_yfp_img[~self.mask_arr] = 0

        # pb corr with constant val
        self.corr_cfp_img = np.asarray([img * (np.sum(np.mean(self.masked_cfp_img[:2], axis=0)/np.sum(img))) \
                                          for img in self.masked_cfp_img])
        self.corr_yfp_img = np.asarray([img * (np.sum(np.mean(self.masked_yfp_img[:2], axis=0)/np.sum(img))) \
                                          for img in self.masked_yfp_img])


    def ch_pic(self):
        """ Shows chsnnels ctrl images and full-frame intensity plots

        """
        v_min = np.min(self.img_raw)
        v_max = np.max(self.img_raw) * .45

        plt.figure(figsize=(12,15))
        ax0 = plt.subplot(221)
        ax0.set_title('Ch. CFP')
        img0 = ax0.imshow(self.cfp_mean_img_raw, cmap=plot.CMaps().cmap_cyan)
        img0.set_clim(vmin=v_min, vmax=v_max)
        div0 = make_axes_locatable(ax0)
        cax0 = div0.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img0, cax=cax0)
        ax0.axis('off')

        ax1 = plt.subplot(222)
        ax1.set_title('Ch. YFP')
        img1 = ax1.imshow(self.yfp_mean_img_raw, cmap=plot.CMaps().cmap_yellow)
        img1.set_clim(vmin=v_min, vmax=v_max)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img1, cax=cax1)
        ax1.axis('off')

        ax2 = plt.subplot(414)
        ax2.plot(np.mean(self.img_raw[:,:,:,0], axis=(1,2)), label='CFP', color='blue')
        ax2.plot(np.mean(self.img_raw[:,:,:,3], axis=(1,2)), label='YFP', color='orange')
        ax2.set_xlabel('Frame num')
        ax2.set_ylabel('Int, a.u.')
        ax2.legend()

        ax3 = plt.subplot(413)
        ax3.hist(self.cfp_mean_img_raw.ravel(), bins=256, alpha=.5, label='CFP', color='blue')
        ax3.hist(self.yfp_mean_img_raw.ravel(), bins=256, alpha=.5, label='YFP', color='orange')
        ax3.set_xlabel('Int, a.u.')
        ax3.set_ylabel('N')
        ax3.legend()

        plt.tight_layout()
        plt.show()

    
    def mask_pic(self):
        """ Shows cell mask

        """
        plt.figure(figsize=(12,15))
        ax0 = plt.subplot(121)
        ax0.set_title('Narrow mask (dentrites contours)')
        ax0.imshow(self.yfp_mean_img_raw, cmap='jet')
        ax0.imshow(ma.masked_where(~self.narrow_proc_mask, self.narrow_proc_mask),
                   cmap='Greys', alpha=.45)
        ax0.axis('off')

        ax1 = plt.subplot(122)
        ax1.set_title('Processes mask')
        ax1.imshow(self.yfp_mean_img_raw, cmap='jet')
        ax1.imshow(ma.masked_where(~self.proc_mask, self.proc_mask),
                   cmap='Greys', alpha=.45)
        ax1.axis('off')

        plt.tight_layout()
        plt.show()
