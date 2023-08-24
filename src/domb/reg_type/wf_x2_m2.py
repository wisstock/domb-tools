"""
Class for wide-field registrations with 2 excitation wl and 2 emission ch.Class for wide-field registrations
with 2 excitation wavelengths and 2 emission pass-band recordings

Optimized for individual neurons imaging

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

from ..utils import masking
from ..utils import plot


class WF_2x_2m():
    def __init__(self, img_path: str, img_name: str, ch_order: dict[str:int],
                 wf_sigma=.5, **kwargs):
        self.img_raw = io.imread(img_path)
        # self.frame_max = filters.gaussian(np.mean(self.img_raw[:,:,:,3], axis=0),
        #                                   sigma=1)
        self.img_name = img_name

        # chennels split
        self.ch0_img = np.asarray([filters.gaussian(frame, sigma=wf_sigma) \
                                   for frame in self.img_raw[:,:,:,0]])  # 435-fp1  DD
        self.ch1_img = np.asarray([filters.gaussian(frame, sigma=wf_sigma) \
                                   for frame in self.img_raw[:,:,:,1]])  # 435-fp2  DA
        self.ch2_img = np.asarray([filters.gaussian(frame, sigma=wf_sigma) \
                                   for frame in self.img_raw[:,:,:,2]])  # 505-fp1  AD
        self.ch3_img = np.asarray([filters.gaussian(frame, sigma=wf_sigma) \
                                   for frame in self.img_raw[:,:,:,3]])  # 505-fp2  AA

        self.ch_list = [self.ch0_img, self.ch1_img, self.ch2_img, self.ch3_img]

        self.fp1_img = self.ch_list[ch_order['fp1']]  # CFP
        self.fp2_img = self.ch_list[ch_order['fp2']]  # YFP

        self.fp1_mean_img_raw = np.mean(self.img_raw[:,:,:,ch_order['fp1']], axis=0)
        self.fp2_mean_img_raw = np.mean(self.img_raw[:,:,:,ch_order['fp2']], axis=0)

        # mask creation
        self.proc_mask = masking.proc_mask(self.fp2_mean_img_raw,
                                           **kwargs)
        self.narrow_proc_mask = masking.proc_mask(self.fp2_mean_img_raw,
                                                  ext_fin_mask=False, **kwargs)

        # img maskings
        self.mask_arr = np.asarray([z+self.proc_mask for z in np.zeros_like(self.fp1_img)], dtype='bool')
        self.masked_fp1_img = np.copy(self.fp1_img)
        self.masked_fp1_img[~self.mask_arr] = 0

        self.masked_fp2_img = np.copy(self.fp2_img)
        self.masked_fp2_img[~self.mask_arr] = 0

        # pb corr with constant val
        self.corr_fp1_img = np.asarray([img * (np.sum(np.mean(self.masked_fp1_img[:2], axis=0)/np.sum(img))) \
                                          for img in self.masked_fp1_img])
        self.corr_fp2_img = np.asarray([img * (np.sum(np.mean(self.masked_fp2_img[:2], axis=0)/np.sum(img))) \
                                          for img in self.masked_fp2_img])


    def hist_pic(self):
        ch0_ctrl = np.mean(self.img_raw[:,:,:,0], axis=0)
        ch1_ctrl = np.mean(self.img_raw[:,:,:,1], axis=0)
        ch2_ctrl = np.mean(self.img_raw[:,:,:,2], axis=0)
        ch3_ctrl = np.mean(self.img_raw[:,:,:,3], axis=0)

        plt.figure(figsize=(20,10))
        ax0 = plt.subplot()
        ax0.hist(ch0_ctrl.ravel(), bins=256, alpha=.5, label='Ch 0 (fp1-435)', color='r')
        ax0.hist(ch1_ctrl.ravel(), bins=256, alpha=.5, label='Ch 1 (fp2-435)', color='g')
        ax0.hist(ch2_ctrl.ravel(), bins=256, alpha=.5, label='Ch 2 (fp1-505)', color='y')
        ax0.hist(ch3_ctrl.ravel(), bins=256, alpha=.5, label='Ch 3 (fp2-505)', color='b')
        ax0.legend()

        plt.show()


    def ch_pic(self):
        """ Shows chsnnels ctrl images and full-frame intensity plots

        """
        v_min = np.min(self.img_raw)
        v_max = np.max(self.img_raw) * .25

        plt.figure(figsize=(12,15))
        ax0 = plt.subplot(221)
        ax0.set_title('Ch. fp1')
        img0 = ax0.imshow(self.fp1_mean_img_raw, cmap=plot.CMaps().cmap_cyan)
        img0.set_clim(vmin=v_min, vmax=v_max)
        div0 = make_axes_locatable(ax0)
        cax0 = div0.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img0, cax=cax0)
        ax0.axis('off')

        ax1 = plt.subplot(222)
        ax1.set_title('Ch. fp2')
        img1 = ax1.imshow(self.fp2_mean_img_raw, cmap=plot.CMaps().cmap_yellow)
        img1.set_clim(vmin=v_min, vmax=v_max)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img1, cax=cax1)
        ax1.axis('off')

        ax2 = plt.subplot(414)
        ax2.plot(np.mean(self.img_raw[:,:,:,0], axis=(1,2)), label='fp1', color='blue')
        ax2.plot(np.mean(self.img_raw[:,:,:,3], axis=(1,2)), label='fp2', color='orange')
        ax2.set_xlabel('Frame num')
        ax2.set_ylabel('Int, a.u.')
        ax2.legend()

        ax3 = plt.subplot(413)
        ax3.hist(self.fp1_mean_img_raw.ravel(), bins=256, alpha=.5, label='fp1', color='blue')
        ax3.hist(self.fp2_mean_img_raw.ravel(), bins=256, alpha=.5, label='fp2', color='orange')
        ax3.set_xlabel('Int, a.u.')
        ax3.set_ylabel('N')
        ax3.legend()

        plt.tight_layout()
        plt.show()

    
    def processes_mask_pic(self):
        """ Shows neuron processes masks

        """
        plt.figure(figsize=(12,15))
        ax0 = plt.subplot(121)
        ax0.set_title('Narrow mask (dentrites contours)')
        ax0.imshow(self.fp2_mean_img_raw, cmap='jet')
        ax0.imshow(ma.masked_where(~self.narrow_proc_mask, self.narrow_proc_mask),
                   cmap='Greys', alpha=.45)
        ax0.axis('off')

        ax1 = plt.subplot(122)
        ax1.set_title('Processes mask')
        ax1.imshow(self.fp2_mean_img_raw, cmap='jet')
        ax1.imshow(ma.masked_where(~self.proc_mask, self.proc_mask),
                   cmap='Greys', alpha=.45)
        ax1.axis('off')

        plt.tight_layout()
        plt.show()
