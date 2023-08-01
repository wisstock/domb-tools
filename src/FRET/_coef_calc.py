"""
Calculation of Unmixing Coefficients and G Parameter

Based on Zal and Gascoigne, 2004, doi: 10.1529/biophysj.103.022087

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage import morphology
from skimage import exposure
from skimage import measure
from skimage import filters
from skimage import io

from ..util import masking



class CrossReg():
    """ Class for 3-cube FRET calibration registration

    """
    def __init__(self, data_path, data_name, exp_list, reg_type, trim_frame=-1):
        self.img_name = data_name
        self.img_type = reg_type
        self.img_path = data_path
        self.img_raw = io.imread(self.img_path)[:trim_frame]

        self.D_exp = exp_list[0]
        self.A_exp = exp_list[1]

        self.DD_img = self.img_raw[:,:,:,0]  # CFP-435  DD
        self.DA_img = self.img_raw[:,:,:,1]  # YFP-435  DA
        self.AD_img = self.img_raw[:,:,:,2]  # CFP-505  AD
        self.AA_img = self.img_raw[:,:,:,3]  # YFP-505  AA

        self.DD_mean_img = np.mean(self.DD_img, axis=0)
        self.DA_mean_img = np.mean(self.DA_img, axis=0)
        self.AD_mean_img = np.mean(self.AD_img, axis=0)
        self.AA_mean_img = np.mean(self.AA_img, axis=0)

        if self.img_type == 'A':
            raw_mask = self.AA_mean_img > filters.threshold_otsu(self.AA_mean_img)
        elif self.img_type == 'D':
            raw_mask = self.DD_mean_img > filters.threshold_otsu(self.DD_mean_img)

        self.mask = morphology.opening(raw_mask, footprint=morphology.disk(10))
        self.mask = morphology.erosion(self.mask, footprint=morphology.disk(5))

        self.DD_img = masking.mask_along_frames(self.DD_img, self.mask)
        self.DA_img = masking.mask_along_frames(self.DA_img, self.mask)
        self.AD_img = masking.mask_along_frames(self.AD_img, self.mask)
        self.AA_img = masking.mask_along_frames(self.AA_img, self.mask)


    def cross_calc(self):
        if self.img_type == 'A':
            self.a_arr = np.asarray([np.mean(self.DA_img[i] / self.AA_img[i]) for i in range(0, self.img_raw.shape[0])])
            self.b_arr = np.asarray([np.mean(self.AD_img[i] / self.AA_img[i]) for i in range(0, self.img_raw.shape[0])])

            self.a = np.mean(self.a_arr)
            self.a_sd = np.std(self.a_arr)

            self.b = np.mean(self.b_arr)
            self.b_sd = np.std(self.b_arr)

            a_df = pd.DataFrame({'ID':np.full(len(self.a_arr), self.img_name),
                                 'type':np.full(len(self.a_arr), self.img_type),
                                 'A_exp':np.full(len(self.a_arr), self.A_exp),
                                 'D_exp':np.full(len(self.a_arr), self.D_exp),
                                 'frame':range(len(self.a_arr)),
                                 'coef':np.full(len(self.a_arr), 'a'),
                                 'val':self.a_arr})
            b_df = pd.DataFrame({'ID':np.full(len(self.b_arr), self.img_name),
                                 'type':np.full(len(self.b_arr), self.img_type),
                                 'A_exp':np.full(len(self.b_arr), self.A_exp),
                                 'D_exp':np.full(len(self.b_arr), self.D_exp),
                                 'frame':range(len(self.b_arr)),
                                 'coef':np.full(len(self.b_arr), 'b'),
                                 'val':self.b_arr})
            self.cross_df = pd.concat([a_df, b_df], ignore_index=True)

            coef_dict = {'ID':[self.img_name, self.img_name],
                         'type':[self.img_type, self.img_type],
                         'A_exp':[self.A_exp, self.A_exp],
                         'D_exp':[self.D_exp, self.D_exp],
                         'coef':['a', 'b'],
                         'val':[self.a, self.b],
                         'sd':[self.a_sd, self.b_sd]}
            self.coef_df = pd.DataFrame(coef_dict)

        elif self.img_type == 'D':
            self.c_arr = np.asarray([np.mean(self.AA_img[i] / self.DD_img[i]) \
                                     for i in range(0, self.img_raw.shape[0])])
            self.d_arr = np.asarray([np.mean(self.DA_img[i] / self.DD_img[i]) \
                                     for i in range(0, self.img_raw.shape[0])])

            self.c = np.mean(self.c_arr)
            self.c_sd = np.std(self.c_arr)

            self.d = np.mean(self.d_arr)
            self.d_sd = np.std(self.d_arr)

            c_df = pd.DataFrame({'ID':np.full(len(self.c_arr), self.img_name),
                                 'type':np.full(len(self.c_arr), self.img_type),
                                 'A_exp':np.full(len(self.c_arr), self.A_exp),
                                 'D_exp':np.full(len(self.c_arr), self.D_exp),
                                 'frame':range(len(self.c_arr)),
                                 'coef':np.full(len(self.c_arr), 'c'),
                                 'val':self.c_arr})
            d_df = pd.DataFrame({'ID':np.full(len(self.d_arr), self.img_name),
                                 'type':np.full(len(self.d_arr), self.img_type),
                                 'A_exp':np.full(len(self.d_arr), self.A_exp),
                                 'D_exp':np.full(len(self.d_arr), self.D_exp),
                                 'frame':range(len(self.d_arr)),
                                 'coef':np.full(len(self.d_arr), 'd'),
                                 'val':self.d_arr})
            self.cross_df = pd.concat([c_df, d_df], ignore_index=True)

            coef_dict = {'ID':[self.img_name, self.img_name],
                         'type':[self.img_type, self.img_type],
                         'A_exp':[self.A_exp, self.A_exp],
                         'D_exp':[self.D_exp, self.D_exp],
                         'coef':['c', 'd'],
                         'val':[self.c, self.d],
                         'sd':[self.c_sd, self.d_sd]}
            self.coef_df = pd.DataFrame(coef_dict)


    def plot_hist(self):
        plt.figure(figsize=(8,8))

        ax0 = plt.subplot(211)
        ax0.hist(self.DD_mean_img.ravel(), bins=256,
                 alpha=.5,label='Ch 0 (CFP-435)', color='r')
        ax0.hist(self.DA_mean_img.ravel(), bins=256,
                 alpha=.5, label='Ch 1 (YFP-435)', color='g')
        ax0.hist(self.AD_mean_img.ravel(), bins=256,
                 alpha=.5, label='Ch 2 (CFP-505)', color='y')
        ax0.hist(self.AA_mean_img.ravel(), bins=256,
                 alpha=.5, label='Ch 3 (YFP-505)', color='b')
        ax0.legend()

        plt.title(f'File {self.img_name}, type {self.img_type}')
        plt.tight_layout()
        plt.show()


    def ff_profile(self):
        plt.figure(figsize=(8,4))

        ax0 = plt.subplot()
        ax0.plot(np.mean(self.DD_img, axis=(1,2)),
                 label='Ch 0 (CFP-435)', color='r')
        ax0.plot(np.mean(self.DA_img, axis=(1,2)),
                 label='Ch 1 (YFP-435)', color='g')
        ax0.plot(np.mean(self.AD_img, axis=(1,2)),
                 label='Ch 2 (CFP-505)', color='y')
        ax0.plot(np.mean(self.AA_img, axis=(1,2)),
                 label='Ch 3 (YFP-505)', color='b')
        ax0.legend()

        plt.title(f'File {self.img_name}, type {self.img_type}')
        plt.tight_layout()
        plt.show()        


    def ch_pic(self):
        int_min = np.min(self.img_raw)
        int_max = np.max(self.img_raw)


        plt.figure(figsize=(10,10))

        ax0 = plt.subplot(221)
        ax0.set_title('DD (Ch.0)')
        img0 = ax0.imshow(self.DD_mean_img, cmap='jet')
        img0.set_clim(vmin=int_min, vmax=int_max)
        div0 = make_axes_locatable(ax0)
        cax0 = div0.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img0, cax=cax0)
        ax0.axis('off')

        ax1 = plt.subplot(222)
        ax1.set_title('DA (Ch.1)')
        img1 = ax1.imshow(self.DA_mean_img, cmap='jet')
        img1.set_clim(vmin=int_min, vmax=int_max)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img1, cax=cax1)
        ax1.axis('off')

        ax2 = plt.subplot(223)
        ax2.set_title('AD (Ch.2)')
        img2 = ax2.imshow(self.AD_mean_img, cmap='jet')
        img2.set_clim(vmin=int_min, vmax=int_max)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img2, cax=cax2)
        ax2.axis('off')

        ax3 = plt.subplot(224)
        ax3.set_title('AA (Ch.3)')
        img3 = ax3.imshow(self.AA_mean_img, cmap='jet')
        img3.set_clim(vmin=int_min, vmax=int_max)
        div3 = make_axes_locatable(ax3)
        cax3 = div3.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img3, cax=cax3)
        ax3.axis('off')

        plt.suptitle(f'File {self.img_name}, type {self.img_type}')
        plt.tight_layout()
        plt.show()