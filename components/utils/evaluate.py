import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from components.utils import angular_resize, cvt_ycrcb2bgr, double2im, sai_io_idx, shave_bd
from components.utils.diff_map import diff_map


def y_upsample(y_chn, ycrcb, a_in, a_out, mp=1):
    """
    Combine estimated Y channel with upsampled CrCb to get upsampled YCrCb.
    :param y_chn: Estimated Y channel, should be output, e.g. (8*8 - 2*2 = 60).
    :param ycrcb: YCrCb input, whose Cr and Cb are to be bicubic upsampled, e.g. (2*2).
    :param a_in: Angular input size, e.g. 2*2.
    :param a_out: Angular output size, e.g. 8*8.
    :return: Upsampled YCrCb.
    """
    a_sz = (ycrcb.shape[0], ycrcb.shape[1])
    s_sz = (ycrcb.shape[2], ycrcb.shape[3])
    chn = ycrcb.shape[4]
    in_sai, out_sai = sai_io_idx(ycrcb.shape[0:2], a_in)
    ycrcb = ycrcb.reshape([a_sz[0] * a_sz[1], s_sz[0], s_sz[1], chn])

    ycrcb_in = ycrcb[in_sai, ...]
    ycrcb_gt = ycrcb[out_sai, ...]
    ycrcb_in = ycrcb_in.reshape([a_in[0], a_in[1], s_sz[0], s_sz[1], chn])

    # Upsample
    ycrcb_up = angular_resize(ycrcb_in, a_out, mp)
    ycrcb_up = ycrcb_up.reshape([a_out[0] * a_out[1], s_sz[0], s_sz[1], chn])
    ycrcb_up = ycrcb_up[out_sai, :, :, :]
    ycrcb_up[:, :, :, 0] = y_chn

    return ycrcb_up, ycrcb_gt


def calc_score_sai(gt, test):
    """
    Calculate PSNR/SSIM score of a frame, could be multi-channels e.g. BGR, YCrCb, or single channel.
    NOTICE: calc_score_sai is without "shave', calc_score_ycrcb_lf is with.
    :param gt: Groudtruth frame with size (s1, s2, chn) or (s1, s2).
    :param test: Test frame with size (s1, s2, chn) or (s1, s2).
    :return: PSNR score, SSIM score.
    """

    psnr_score = []
    ssim_score = []

    for chn in range(gt.shape[2]):
        psnr_score.append(psnr(gt[:, :, chn], test[:, :, chn]))
        ssim_score.append(ssim(gt[:, :, chn], test[:, :, chn]))

    return np.mean(psnr_score), np.mean(ssim_score)


def calc_score_ycrcb_lf(y_chn, ycrcb, bgr, a_in, a_out, render_diff=False):
    """
    Calculate PSNR/SSIM score of single sample, scores will be calculated on BGR converted from YCrCb.
    :param y_chn: the estimated Y channel.
    :param ycrcb: the YCrCb groundtruth, providing Cr and Cb to be upsampled..
    :param bgr: the bgr groundtruth, real groundtruth to be compared.
    :param a_in: Angular input size, (a1, a1).
    :param a_out: Angular output size, (a2, a2).
    :return: PSNR score, SSIM score.
    :param render_diff: Path to save diff maps. If False, don't render diff maps.
    """
    ycrcb_up, ycrcb_gt = y_upsample(y_chn, ycrcb, a_in, a_out)
    in_sai, out_sai = sai_io_idx(bgr.shape[0:2], a_in)
    bgr = bgr.reshape([bgr.shape[0]*bgr.shape[1], bgr.shape[2], bgr.shape[3], bgr.shape[4]])[out_sai, ...]

    # Calculate
    psnr_a_list = []
    ssim_a_list = []

    for idx_a in range(bgr.shape[0]):
        bgr_up = cvt_ycrcb2bgr(ycrcb_up[idx_a])
        bgr_up = double2im(bgr_up)
        bgr_gt = bgr[idx_a]

        off = np.array(bgr_gt.shape[0:2]) - np.array(bgr_up.shape[0:2])

        bgr_gt = shave_bd(bgr_gt, 22)
        bgr_up = bgr_up[22:-22+off[0], 22:-22+off[1]]

        psnr_a, ssim_a = calc_score_sai(bgr_gt, bgr_up)
        psnr_a_list.append(np.mean(psnr_a))
        ssim_a_list.append(np.mean(ssim_a))

        if render_diff:
            diff_color = diff_map(bgr_gt, bgr_up)
            cv2.imwrite(render_diff % (idx_a+1), diff_color)

    return np.mean(psnr_a_list), np.mean(ssim_a_list)


def postprocess(y):
    """
    Process the predicted result to eliminate values more than 1 or less than 0
    :param y: Raw predicted result
    :return: Post processed result
    """
    y[y < 0] = 0
    y[y > 1] = 1
    return y
