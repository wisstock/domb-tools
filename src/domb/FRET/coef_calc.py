"""
Calculation of Unmixing Coefficients and G Parameter

Based on Zal and Gascoigne, 2004, doi: 10.1529/biophysj.103.022087

"""

import numpy as np
from numpy import ma
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage import morphology
from skimage import filters
from skimage import io

from ..util import masking


# crosstalk estimation
class CrossReg():
    """ Class for one 3-cube FRET method crosstalk calibration registration

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
            self.cross_raw_df = pd.concat([a_df, b_df], ignore_index=True)

            coef_dict = {'ID':[self.img_name, self.img_name],
                         'type':[self.img_type, self.img_type],
                         'A_exp':[self.A_exp, self.A_exp],
                         'D_exp':[self.D_exp, self.D_exp],
                         'coef':['a', 'b'],
                         'val':[self.a, self.b],
                         'sd':[self.a_sd, self.b_sd]}
            self.cross_df = pd.DataFrame(coef_dict)

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
            self.cross_raw_df = pd.concat([c_df, d_df], ignore_index=True)

            coef_dict = {'ID':[self.img_name, self.img_name],
                         'type':[self.img_type, self.img_type],
                         'A_exp':[self.A_exp, self.A_exp],
                         'D_exp':[self.D_exp, self.D_exp],
                         'coef':['c', 'd'],
                         'val':[self.c, self.d],
                         'sd':[self.c_sd, self.d_sd]}
            self.cross_df = pd.DataFrame(coef_dict)


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


class CrossRegSet():
    """ Class processing set of 3-cube FRET method crosstalk calibration registrations

    Requires sets of (A) and (D) registrations

    """
    def __init__(self, data_path, donor_reg_dict, acceptor_reg_dict, trim_frame=-1):
        self.donor_reg = donor_reg_dict
        self.acceptor_reg = acceptor_reg_dict


        self.donor_reg_list = []
        for reg_name in self.donor_reg.keys():
            name_path = data_path + f'{reg_name}.tif'
            self.donor_reg_list.append(CrossReg(data_path=name_path,
                                        data_name=reg_name,
                                        exp_list=self.donor_reg[reg_name],
                                        reg_type='D',
                                        trim_frame=trim_frame))
        self.acceptor_reg_list = []
        for reg_name in self.acceptor_reg.keys():
            name_path = data_path + f'{reg_name}.tif'
            self.acceptor_reg_list.append(CrossReg(data_path=name_path,
                                                   data_name=reg_name,
                                                   exp_list=self.acceptor_reg[reg_name],
                                                   reg_type='A',
                                                   trim_frame=trim_frame))


    def get_abcd(self, show_pic=False):
        """ Create data frame with calculated crosstalc coeficients for each calibration registration

        """

        self.cross_raw_df = pd.DataFrame(columns=['ID',  # df with raw results
                                                   'type',
                                                   'A_exp',
                                                   'D_exp',
                                                   'frame',
                                                   'coef',
                                                   'val'])

        self.cross_df = pd.DataFrame(columns=['ID',
                                                  'type',
                                                  'A_exp',
                                                  'D_exp',
                                                  'coef',
                                                  'val',
                                                  'sd'])

        for reg in self.acceptor_reg_list + self.donor_reg_list:
            if show_pic:
                reg.ch_pic()
            reg.cross_calc()
            self.cross_raw_df = pd.concat([self.cross_raw_df, reg.cross_df], ignore_index=True)
            self.cross_df = pd.concat([self.cross_df, reg.cross_df], ignore_index=True)

        return self.cross_val_df
    

# G parameter estimation
class GReg():
    def __init__(self, img_name, raw_path, bleach_path, bleach_frame, bleach_exp, A_exp, D_exp, coef_list, trim_frame=-1):
        self.img_name = img_name
        self.img_raw = io.imread(raw_path)[:trim_frame]
        self.img_bleach = io.imread(bleach_path)[:trim_frame]

        self.bleach_frame = bleach_frame
        self.bleach_exp = bleach_exp
        self.A_exp = A_exp
        self.D_exp = D_exp

        self.a = coef_list[0]
        self.b = coef_list[1]
        self.c = coef_list[2]
        self.d = coef_list[3]


        self.DD_img = self.img_raw[:,:,:,0]  # CFP-435  DD
        self.DA_img = self.img_raw[:,:,:,1]  # YFP-435  DA
        self.AD_img = self.img_raw[:,:,:,2]  # CFP-505  AD
        self.AA_img = self.img_raw[:,:,:,3]  # YFP-505  AA

        self.DD_img_post = self.img_bleach[:,:,:,0]
        self.DA_img_post = self.img_bleach[:,:,:,1]
        self.AD_img_post = self.img_bleach[:,:,:,2]
        self.AA_img_post = self.img_bleach[:,:,:,3]


        self.DD_mean_img = np.mean(self.DD_img, axis=0)
        self.DA_mean_img = np.mean(self.DA_img, axis=0)
        self.AD_mean_img = np.mean(self.AD_img, axis=0)
        self.AA_mean_img = np.mean(self.AA_img, axis=0)

        raw_mask = self.AA_mean_img > filters.threshold_otsu(self.AA_mean_img)

        self.mask = morphology.opening(raw_mask, footprint=morphology.disk(10))
        self.mask = morphology.erosion(self.mask, footprint=morphology.disk(5))

        self.DD_img = masking.mask_along_frames(self.DD_img, self.mask)
        self.DA_img = masking.mask_along_frames(self.DA_img, self.mask)
        self.AD_img = masking.mask_along_frames(self.AD_img, self.mask)
        self.AA_img = masking.mask_along_frames(self.AA_img, self.mask)

        self.DD_img_post = masking.mask_along_frames(self.DD_img_post, self.mask)
        self.DA_img_post = masking.mask_along_frames(self.DA_img_post, self.mask)
        self.AD_img_post = masking.mask_along_frames(self.AD_img_post, self.mask)
        self.AA_img_post = masking.mask_along_frames(self.AA_img_post, self.mask)


    def G_calc(self):
        self.Fc_raw = self.__Fc_img(dd_img=self.DD_img,
                                    da_img=self.DA_img,
                                    aa_img=self.AA_img,
                                    a=self.a, b=self.b, c=self.c, d=self.d,
                                    mask=self.mask)
        self.Fc_post = self.__Fc_img(dd_img=self.DD_img_post,
                                     da_img=self.DA_img_post, aa_img=self.AA_img_post,
                                     a=self.a, b=self.b, c=self.c, d=self.d,
                                     mask=self.mask)
        
        self.G_img = self.__G_img(Fc_pre_img=self.Fc_raw, Fc_post_img=self.Fc_post,
                                  dd_pre_img=self.DD_img, dd_post_img=self.DD_img_post,
                                  mask=self.mask)
        self.G_profile = np.mean(self.G_img, axis=(1,2), where=self.mask)
        self.G = np.mean(self.G_profile[10:-10])
        self.G_sd = np.std(self.G_profile[10:-10])
        print(f'G={self.G}+/-{self.G_sd}')


    @staticmethod
    def __Fc_img(dd_img, da_img, aa_img, a, b, c, d):
        Fc_img = []
        for frame_num in range(dd_img.shape[0]):
            DD_frame = dd_img[frame_num]
            DA_frame = da_img[frame_num]
            AA_frame = aa_img[frame_num]

            Fc_frame = DA_frame - a*(AA_frame - c*DD_frame) - d*(DD_frame - b*AA_frame)
            Fc_img.append(Fc_frame)

        return np.asarray(Fc_img)
    

    @staticmethod
    def __G_img(Fc_pre_img, Fc_post_img, dd_pre_img, dd_post_img, mask):
        G_img = []
        for frame_num in range(Fc_pre_img.shape[0]):
            Fc_pre_frame = ma.masked_where(~mask, Fc_pre_img[frame_num])
            Fc_post_frame = ma.masked_where(~mask, Fc_post_img[frame_num])
            DD_pre_frame = ma.masked_where(~mask, dd_pre_img[frame_num])
            DD_post_frame = ma.masked_where(~mask, dd_post_img[frame_num])

            G_frame = (Fc_pre_frame - Fc_post_frame) / (DD_post_frame - DD_pre_frame)
            G_img.append(G_frame)
        
        return np.asarray(G_img)


    def Fc_plot(self, raw_frames=-1, post_frames=-1):
        Fc_raw_mean = np.mean(self.Fc_raw[:raw_frames], axis=0)
        Fc_post_mean = np.mean(self.Fc_post[:post_frames], axis=0)

        plt.figure(figsize=(10,5))

        ax0 = plt.subplot(221)
        ax0.set_title('Fc pre mean')
        img0 = ax0.imshow(ma.masked_where(~self.mask, Fc_raw_mean), cmap='jet')
        # img0.set_clim(vmin=int_min, vmax=int_max)
        div0 = make_axes_locatable(ax0)
        cax0 = div0.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img0, cax=cax0)
        ax0.axis('off')

        ax1 = plt.subplot(222)
        ax1.set_title('Fc raw FF int profile')
        ax1.plot(np.mean(self.Fc_raw, axis=(1,2), where=self.mask),
                 label='Fc', color='r')
        ax1.plot(np.mean(self.DA_img, axis=(1,2), where=self.mask),
                 label='DA', color='r', linestyle='--')
        ax1.legend()

        ax2 = plt.subplot(223)
        ax2.set_title('Fc post mean')
        img2 = ax2.imshow(ma.masked_where(~self.mask, Fc_post_mean), cmap='jet')
        # img0.set_clim(vmin=int_min, vmax=int_max)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img2, cax=cax2)
        ax2.axis('off')

        ax3 = plt.subplot(224)
        ax3.set_title('Fc post FF int profile')
        ax3.plot(np.mean(self.Fc_post, axis=(1,2), where=self.mask),
                 label='Fc', color='r')
        ax3.plot(np.mean(self.DA_img_post, axis=(1,2), where=self.mask),
                 label='DA', color='r', linestyle='--')
        ax3.legend()

        plt.suptitle(f'File {self.img_name}')
        plt.tight_layout()
        plt.show()
        

    def G_plot(self):
        G_mean = ma.masked_where(~self.mask, np.mean(self.G_img, axis=0))
        G_median = np.median(self.G_profile)

        plt.figure(figsize=(10,5))

        ax0 = plt.subplot(121)
        ax0.set_title('G mean')
        img0 = ax0.imshow(G_mean, cmap='jet')
        # img0.set_clim(vmin=int_min, vmax=int_max)
        div0 = make_axes_locatable(ax0)
        cax0 = div0.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img0, cax=cax0)
        ax0.axis('off')

        ax1 = plt.subplot(122)
        ax1.set_title('G FF pofile')
        ax1.plot(self.G_profile, color='r', label='profile')
        ax1.hlines(y=G_median, xmin=0, xmax=self.G_profile.shape[0],
                   linestyles='--', color='r', label='median')
        ax1.legend()

        plt.suptitle(f'File {self.img_name}')
        plt.tight_layout()
        plt.show()        


    def pre_post_plot(self):
        plt.figure(figsize=(10,5))

        ax0 = plt.subplot(221)
        ax0.set_title('DD (Ch.0)')
        ax0.plot(np.mean(self.DD_img, axis=(1,2), where=self.mask),
                 label='pre', color='r')
        ax0.plot(np.mean(self.DD_img_post, axis=(1,2), where=self.mask),
                 label='post', color='r', linestyle='--')
        ax0.legend()

        ax1 = plt.subplot(222)
        ax1.set_title('DA (Ch.1)')
        ax1.plot(np.mean(self.DA_img, axis=(1,2), where=self.mask),
                 label='pre', color='g')
        ax1.plot(np.mean(self.DA_img_post, axis=(1,2), where=self.mask),
                 label='post', color='g', linestyle='--')
        ax1.legend()

        ax2 = plt.subplot(223)
        ax2.set_title('AD (Ch.2)')
        ax2.plot(np.mean(self.AD_img, axis=(1,2), where=self.mask),
                 label='pre', color='y')
        ax2.plot(np.mean(self.AD_img_post, axis=(1,2), where=self.mask),
                 label='post', color='y', linestyle='--')
        ax2.legend()

        ax3 = plt.subplot(224)
        ax3.set_title('AA (Ch.3)')
        ax3.plot(np.mean(self.AA_img, axis=(1,2), where=self.mask),
                 label='pre', color='b')
        ax3.plot(np.mean(self.AA_img_post, axis=(1,2), where=self.mask),
                 label='post', color='b', linestyle='--')
        ax3.legend()
        
        plt.suptitle(f'File {self.img_name}')
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

        plt.suptitle(f'File {self.img_name}')
        plt.tight_layout()
        plt.show()


class GRegSet():
        def __init__(self, data_path, tandem_reg_dict, abcd_list):
            self.tandem_reg_list = []
            for reg_name in tandem_reg_dict.keys():
                reg_params = tandem_reg_dict[reg_name]
                raw_path = data_path + f'{reg_params[0]}.tif'
                bleach_path = data_path + f'{reg_params[1]}.tif'
                self.tandem_reg_list.append(GReg(img_name=reg_name,
                                                 raw_path=raw_path,
                                                 bleach_path=bleach_path,
                                                 bleach_frame=reg_params[2],
                                                 bleach_exp=reg_params[3],
                                                 A_exp=reg_params[4],
                                                 D_exp=reg_params[5],
                                                 coef_list=abcd_list))


        def get_G(self, show_pic=False):
            """ Create data frame with calculated crosstalc coeficients for each calibration registration

            """
            pass