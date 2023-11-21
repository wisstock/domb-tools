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


class wf_x2_m2():
    def __init__(self, img_path:str, img_name:str,
                ch_order:dict[str:int], use_gauss:bool=False, wf_sigma:float=.5, border_crop:int=0,
                **kwargs):
        """ Class is designed to store experiment data from wide-field imaging
        using two different excitation wavelengths and two emission channels (__2 eXcitation & 2 eMission__).
        It's specifically optimized for the results of individual neuron imaging.

        Parameters
        ----------
        img_path: str
            path to the image series TIFF file
        img_name: str
            recording name
        ch_order: dict
            indexes of 1st and 2nd fluorescence proteins channels,
            if fluorescence proteins doesn't overlap (`{'fp1': index, 'fp2': index}`)
        use_gauss: boolean, optional
            if `True` Gaussian filter will be applied to input image series
        wf_sigma: float, optional
            sigma value for Gaussian filter applied to input image series
        border_crop: int (0)
            image crop size in pixels,
            this amount of pixels will be deleted from sides on each frame
            
        Attributes
        ----------
        img_raw: ndarray [t,x,y,c]
            raw input image series
        img_name: str
            registration name
        fp1_img: ndarray [t,x,y]
            1st fluorescence protein image series
            with photobleaching correction and Gaussian filtering (if `use_gauss` is `True`)
        fp1_img_raw: ndarray [t,x,y]
            1st fluorescence protein image series,
            without photobleaching correction and Gaussian filtering
        fp1_img_corr: ndarray [t,x,y]
            1st fluorescence image series
            with photobleaching correction
        fp1_mean_img_raw: ndarray [x,y]
            1st fluorescence protein pixel-wise mean image
        fp2_img: ndarray [t,x,y]
            2nd fluorescence protein image series,
            with photobleaching correction and Gaussian filtering (if `use_gauss` is `True`) 
        fp2_img_raw: ndarray [t,x,y]
            2nd fluorescence protein image series,
            without photobleaching correction and Gaussian filtering
        fp2_img_corr: ndarray [t,x,y]
            2nd fluorescence protein image series
            with photobleaching correction
        fp2_mean_img_raw: ndarray [x,y]
            2nd fluorescence protein pixel-wise mean image
        proc_mask: ndarray [x,y]
            cell processes boolean mask, extended (created with `utils.masking.proc_mask()`)
        narrow_proc_mask: ndarray [x,y]
            cell processes boolean mask, unextended (created with `utils.masking.proc_mask()`)


        """
        self.img_raw = io.imread(img_path)
        if border_crop !=0:  # optional crop of image series
            y,x = self.img_raw.shape[1:3]
            self.img_raw = self.img_raw[:,border_crop:y-border_crop,border_crop:x-border_crop,:]

        self.img_name = img_name
        self.ch_order = ch_order
        self.gauss_sigma = gauss_sigma

        # primary image extraction
        self.fp1_img_raw = self.img_raw[:,:,:,ch_order['fp1']]
        self.fp2_img_raw = self.img_raw[:,:,:,ch_order['fp2']]

        self.fp1_mean_img_raw = np.mean(self.fp1_img_raw, axis=0)
        self.fp2_mean_img_raw = np.mean(self.fp2_img_raw, axis=0)

        # mask creation
        self.proc_mask = masking.proc_mask(self.fp2_mean_img_raw,
                                           ext_fin_mask=True, **kwargs)
        self.narrow_proc_mask = masking.proc_mask(self.fp2_mean_img_raw,
                                                  ext_fin_mask=False, **kwargs)

        # photobleaching correction and bluring
        self.fp1_img_corr,self.fb1_pbc,self.fb1_pb_r = masking.pb_exp_correction(input_img=self.fp1_img_raw,
                                                                                 mask=self.proc_mask)
        
        self.fp1_back_mean = np.mean(self.fp1_img_corr, axis=(1,2), where=~self.proc_mask)
        self.fp1_back_mean = self.fp1_back_mean.reshape(-1, 1, 1)
        self.fp1_img_corr = self.fp1_img_corr - self.fp1_back_mean
        self.fp2_img_corr,self.fb2_pbc,self.fb2_pb_r = masking.pb_exp_correction(input_img=self.fp2_img_raw,
                                                                                 mask=self.proc_mask)
        self.fp2_back_mean = np.mean(self.fp2_img_corr, axis=(1,2), where=~self.proc_mask)
        self.fp2_back_mean = self.fp2_back_mean.reshape(-1, 1, 1)
        self.fp2_img_corr = self.fp2_img_corr - self.fp2_back_mean

        if use_gauss:
            self.fp1_img = np.asarray([filters.gaussian(frame, sigma=self.gauss_sigma) \
                                       for frame in self.fp1_img_corr])
            self.fp2_img = np.asarray([filters.gaussian(frame, sigma=self.gauss_sigma) \
                                       for frame in self.fp2_img_corr])
        else:
            self.fp1_img = self.fp1_img_corr
            self.fp2_img = self.fp2_img_corr


        # # img maskings
            # mask_arr: ndarray [t,x,y]
            #     cell processes mask array, series of `proc_mask`
            # masked_fp1_img: ndarray [t,x,y]
            #     1st fluorescence protein image series masked frame-by-frame with `proc_mask`,
            #     out of mask pixels set as 0    
            # masked_fp2_img: ndarray [t,x,y]
            #     2nd fluorescence protein image series masked frame-by-frame with `proc_mask`,
            #     out of mask pixels set as 0
        # self.mask_arr = np.asarray([z+self.proc_mask for z in np.zeros_like(self.fp1_img)], dtype='bool')
        # self.masked_fp1_img = np.copy(self.fp1_img)
        # self.masked_fp1_img[~self.mask_arr] = 0

        # self.masked_fp2_img = np.copy(self.fp2_img)
        # self.masked_fp2_img[~self.mask_arr] = 0


        # PB CORR VARIANTS
        # pb corr with constant val, old version with zeros background
        # self.corr_fp1_img = np.asarray([img * (np.sum(np.mean(self.masked_fp1_img[:2], axis=0)/np.sum(img))) \
        #                                   for img in self.masked_fp1_img])
        # self.corr_fp2_img = np.asarray([img * (np.sum(np.mean(self.masked_fp2_img[:2], axis=0)/np.sum(img))) \
        #                                   for img in self.masked_fp2_img])
        
        # pb corr with constant val, without zeros background
        # self.corr_fp1_img = np.asarray([self.fp1_img[img_i] * (np.sum(np.mean(self.masked_fp1_img[:2], axis=0)/np.sum(self.masked_fp1_img[img_i]))) \
        #                                   for img_i in range(self.masked_fp1_img.shape[0])])
        # self.corr_fp2_img = np.asarray([self.fp2_img[img_i] * (np.sum(np.mean(self.masked_fp2_img[:2], axis=0)/np.sum(self.masked_fp2_img[img_i]))) \
        #                                   for img_i in range(self.masked_fp2_img.shape[0])])


    def get_all_channels(self):
        """ Returns all individual channels blurred with Gaussian filter (with sigma `wf_sigma`).
        
        Could be useful for FRET calculation.

        Returns
        -------
        ch0_img: ndarray [t,x,y]
            image series for channel 0
        ch1_img: ndarray [t,x,y]
            image series for channel 0
        ch2_img: ndarray [t,x,y]
            image series for channel 0
        ch3_img: ndarray [t,x,y]
            image series for channel 0

        """
        # chennels split
        ch0_img = np.asarray([filters.gaussian(frame, sigma=self.gauss_sigma) \
                              for frame in self.img_raw[:,:,:,0]])  # 435-fp1  DD
        ch1_img = np.asarray([filters.gaussian(frame, sigma=self.gauss_sigma) \
                              for frame in self.img_raw[:,:,:,1]])  # 435-fp2  DA
        ch2_img = np.asarray([filters.gaussian(frame, sigma=self.gauss_sigma) \
                              for frame in self.img_raw[:,:,:,2]])  # 505-fp1  AD
        ch3_img = np.asarray([filters.gaussian(frame, sigma=self.gauss_sigma) \
                              for frame in self.img_raw[:,:,:,3]])  # 505-fp2  AA

        return ch0_img, ch1_img, ch2_img, ch3_img

    def ch_pic(self):
        """ Plotting of registration control image

        """
        v_min = np.min(self.img_raw)
        v_max = np.max(self.img_raw) * .5

        fp1_der_w, fp1_der_p = masking.series_derivate(input_img=self.fp1_img_corr,
                                                       mask=self.proc_mask,
                                                       left_win=2, space=4, right_win=2)
        fp2_der_w, fp2_der_p = masking.series_derivate(input_img=self.fp2_img_corr,
                                                       mask=self.proc_mask,
                                                       left_win=2, space=4, right_win=2)

        plt.figure(figsize=(10,10))
        ax0 = plt.subplot(231)
        ax0.set_title('fp1')
        ax0.imshow(self.fp1_mean_img_raw, cmap='jet')
        ax0.axis('off')

        ax1 = plt.subplot(232)
        ax1.set_title('fp2')
        img1 = ax1.imshow(self.fp2_mean_img_raw, cmap='jet')
        img1.set_clim(vmin=v_min, vmax=v_max)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img1, cax=cax1)
        ax1.axis('off')

        ax4 = plt.subplot(233)
        ax4.imshow(self.fp2_mean_img_raw, cmap='jet')
        ax4.imshow(ma.masked_where(~self.narrow_proc_mask, self.narrow_proc_mask),
                   cmap='Greys', alpha=.4)
        ax4.imshow(ma.masked_where(~self.proc_mask, self.proc_mask),
                   cmap='Greys', alpha=.25)
        ax4.set_title('processes mask')
        ax4.axis('off')

        ax2 = plt.subplot(413)
        ax2.plot(np.mean(self.fp1_img_raw, axis=(1,2), where=self.proc_mask),
                 label='fp1 raw', color='k')
        ax2.plot(np.mean(self.fp1_img_corr, axis=(1,2), where=self.proc_mask),
                 label='fp1 corrected', linestyle='--', color='k')
        ax2.plot(np.mean(self.fp2_img_raw, axis=(1,2), where=self.proc_mask),
                 label='fp2 raw', color='r')
        ax2.plot(np.mean(self.fp2_img_corr, axis=(1,2), where=self.proc_mask),
                 label='fp2 corrected', linestyle='--', color='r')
        ax2.set_xlabel('Frame num')
        ax2.set_ylabel('Int, a.u.')
        ax2.set_title('Int. profile')
        ax2.legend()

        ax3 = plt.subplot(414)
        ax3.plot(fp2_der_p, linestyle='--', color='r')
        ax3.plot(fp1_der_p, linestyle='--', color='k')
        ax3.plot(fp2_der_w, color='r')
        ax3.plot(fp1_der_w, color='k')
        ax3.set_xlabel('Frame num')
        ax3.set_ylabel('Norm. der.')
        ax3.set_title('Der. profile')
        ax3.legend()

        plt.suptitle(self.img_name)
        plt.tight_layout()
        plt.show()

    
    def processes_mask_pic(self):
        """ Plotting of cell processes mask (narrow and extended)
        overlay with control fluorescence image (`fp2_mean_img_raw`).

        """
        plt.figure(figsize=(10,190))
        ax0 = plt.subplot()
        ax0.imshow(self.fp2_mean_img_raw, cmap='jet')
        ax0.imshow(ma.masked_where(~self.narrow_proc_mask, self.narrow_proc_mask),
                   cmap='Greys', alpha=.4)
        ax0.imshow(ma.masked_where(~self.proc_mask, self.proc_mask),
                   cmap='Greys', alpha=.25)
        ax0.axis('off')

        plt.tight_layout()
        plt.show()


    def hist_pic(self):
        """ Plotting of histogram for individual channels
        (for channel mean intensity frame).
        
        """

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