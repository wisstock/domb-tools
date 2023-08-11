"""
Class for red-green maskig of co-recording WT and mutant protein

Optimized for individual neurons imaging

Requires WS_2x_2m type as input

"""



def up_mask_calc(input_img_series, ctrl_img, base_frames=5, app_start=7, app_win=5):
    ref_img_series = filters.gaussian(input_img_series, sigma=1.25, channel_axis=0)

    img_base = np.mean(ref_img_series[:base_frames], axis=0)
    img_max = np.mean(ref_img_series[app_start:app_start+app_win], axis=0)

    img_diff = img_max - img_base
    img_diff = img_diff/np.max(np.abs(img_diff))

    diff_sd = np.std(ma.masked_where(~mask, img_diff))
    up_mask = img_diff > diff_sd * 2

    up_mask_filt = morphology.opening(up_mask, footprint=morphology.disk(2))
    up_mask_filt = morphology.dilation(up_mask_filt, footprint=morphology.disk(1))
    up_label = measure.label(up_mask_filt)


    plt.figure(figsize=(20,20))
    ax0 = plt.subplot(131)
    ax0.set_title('Differential img')
    ax0.imshow(img_diff, cmap=cmap_red_green, vmax=1, vmin=-1)
    ax0.axis('off')

    ax1 = plt.subplot(132)
    ax1.set_title('Up regions')
    ax1.imshow(img_yfp_ctrl)
    ax1.imshow(ma.masked_where(~up_mask, up_mask), alpha=0.5, cmap=cmap_red)
    ax1.axis('off')

    ax2 = plt.subplot(133)
    ax2.set_title(f'Up mask labels ({up_label.max()} regions)')
    ax2.imshow(ctrl_img)
    ax2.imshow(ma.masked_where(~up_mask_filt, up_label), alpha=0.5, cmap='bwr')
    ax2.arrow(550,645,-30,0,width=10, alpha=0.25, color='white')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    return up_mask_filt, up_label, img_diff


def up_mask_connection(input_wt_mask, input_mutant_mask):
    wt_label, wt_num = ndi.label(input_wt_mask)

    sums = ndi.sum(input_mutant_mask, wt_label, np.arange(wt_num+1))
    connected = sums > 0
    debris_mask = connected[wt_label]

    fin_mask = np.copy(input_wt_mask)
    fin_mask[~debris_mask] = 0

    fin_label, fin_num = ndi.label(fin_mask)

    return fin_mask, fin_label