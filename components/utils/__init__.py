import functools
import os
from multiprocessing.pool import Pool
from os import path
import cv2
import numpy as np
from keras import backend as K


def get_dir(*pth):
    """join dir pth and create it when it doesn't exist"""
    pth_ = path.join(*pth)
    if not path.exists(pth_):
        os.makedirs(pth_)
    return pth_


def im2double(im, dtype='float32'):
    info = np.iinfo(im.dtype)
    return im.astype(dtype) / info.max


def double2im(im, dtype='uint16'):
    info = np.iinfo(dtype)
    return (im * info.max).astype(dtype)


def shave_bd(img, bd):
    """
    Shave border area of spatial views. A common operation in SR.
    :param img:
    :param bd:
    :return:
    """
    return img[bd:-bd, bd:-bd, :]


def sai_io_idx(a_out, a_in):
    idx_mat = np.array(range(a_out[0] * a_out[1])).astype(np.int).reshape(a_out)
    idx_map = np.ones(idx_mat.shape).astype(np.int)
    step = (np.array(a_out) - np.array(a_in)) // (np.array(a_in) - 1) + 1

    for y in range(0, a_out[0], step[0]):
        for x in range(0, a_out[1], step[1]):
            idx_map[y, x] = -1

    idx_in = idx_mat[idx_map == -1]
    idx_out = idx_mat[idx_map != -1]
    return idx_in, idx_out


def lf_raw2bgr(raw_lf, a_sz, a_preserve):
    """
    Convert raw LF to structured BGR color matrix.
    :param raw_lf: Raw image matrix (a*n, a*n, 3)
    :param a_sz: Size of angular dims
    :param a_preserve: Number of preserved angular views
    :return: BGR color matrix (a, a, n, n, 3)
    """
    s_sz = (raw_lf.shape[0:2] / np.array(a_sz)).astype(np.int32)
    bgr_lf = np.zeros([
        a_sz[0], a_sz[1],
        s_sz[0], s_sz[1], 3,
    ]).astype(raw_lf.dtype)
    for ay in range(a_sz[0]):
        for ax in range(a_sz[1]):
            bgr_lf[ay, ax, :, :, :] = raw_lf[ay::a_sz[0], ax::a_sz[1], 0:3]
    off = ((np.array(a_sz) - np.array(a_preserve)) / 2).astype(np.int32)  # Preserve middle area and cast out margin
    bgr_lf = bgr_lf[off[0]:-off[0], off[1]:-off[1], :, :, :]  # [3:11, 3:11, 3, :, :]
    return bgr_lf


def lf_bgr2ycrcb(bgr_lf):
    """Convert structured BGR color matrix to structured YCrCb color matrix."""
    return lf_cvt(bgr_lf, cv2.COLOR_BGR2YCR_CB)


def matlab_rgb2ycbcr(rgb):
    """
    the same as matlab rgb2ycbcr
    :param rgb: input [0, 255] or [0, 1]
    :return: output [0, 255] or [0, 1]
    """
    in_img_type = rgb.dtype
    rgb = rgb.astype(np.float64)
    if in_img_type != np.uint8:
        rgb *= 255.
    m = np.array([[65.481, 128.553, 24.966],
                    [-37.797, -74.203, 112],
                    [112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0]*shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    # ycbcr = np.clip(ycbcr, 0, 255)
    if in_img_type == np.uint8:
        ycbcr = ycbcr.round()
    else:
        ycbcr /= 255.

    return ycbcr.reshape(shape).astype(in_img_type)
 

def matlab_ycbcr2rgb(ycbcr):
    """
    the same as matlab ycbcr2rgb
    :param rgb: input [0, 255] or [0, 1]
    :return: output [0, 255] or [0, 1]
    """
    in_img_type = ycbcr.dtype
    ycbcr = ycbcr.astype(np.float64)
    if in_img_type != np.uint8:
        ycbcr *= 255.
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])

    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0]*shape[1], 3))

    rgb = np.copy(ycbcr)
    rgb[:, 0] -= 16.
    rgb[:, 1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    rgb = np.clip(rgb, 0, 255)
    if in_img_type == np.uint8:
        rgb = rgb.round()
    else:
        rgb /= 255.

    return rgb.reshape(shape).astype(in_img_type)


def lf_cvt(src, code):
    """Convert structured matrix to other color space."""
    ycrcb_lf = np.zeros(src.shape).astype(src.dtype)
    for ay in range(src.shape[0]):
        for ax in range(src.shape[1]):
            img = src[ay, ax, :, :, :]
            ycrcb_lf[ay, ax, :, :, :] = cv2.cvtColor(img, code)
    return ycrcb_lf


def lf_matlab_cvt(src, code):
    """Convert structured matrix to other color space."""
    ycrcb_lf = np.zeros(src.shape).astype(src.dtype)
    for ay in range(src.shape[0]):
        for ax in range(src.shape[1]):
            img = src[ay, ax, :, :, :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ycrcb_lf[ay, ax, :, :, :] = matlab_rgb2ycbcr(img)
    return ycrcb_lf


def cvt_ycrcb2bgr(ycrcb):
    """
    Convert image from YCrCb to BGR, fixing the overflow problem.
    :param ycrcb: YCrCb image, should be [0, 1].
    :return: BGR image
    """
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    bgr[bgr < 0] = 0
    bgr[bgr > 1] = 1
    return bgr


def angular_resize(lf, sz_a_dst, mp=1):
    """
    Resize angularly.
    :param lf: LF matrix (a1, a1, s, s, chn)
    :param sz_a_dst: Destination angular size (a2, a2)
    :return: Resized LF matrix (a2, a2, s, s, chn)
    """
    a_sz = (lf.shape[0], lf.shape[1])
    s_sz = (lf.shape[2], lf.shape[3])
    chn = lf.shape[4]
    lf = lf.transpose([2, 3, 0, 1, 4]).reshape([s_sz[0] * s_sz[1], a_sz[0], a_sz[1], chn])

    lf_up = []
    target = functools.partial(cv2.resize,
                               dsize=(sz_a_dst[0], sz_a_dst[1]), interpolation=cv2.INTER_LINEAR)
    if mp > 1:
        with Pool(mp) as pool:
            lf_up = pool.starmap(target, zip(lf))
    else:
        for i in range(lf.shape[0]):
            up = cv2.resize(lf[i], dsize=(sz_a_dst[0], sz_a_dst[1]), interpolation=cv2.INTER_LINEAR)
            lf_up.append(up)
    lf_up = np.array(lf_up)
    lf_up = lf_up.transpose([1, 2, 0, 3]).reshape([sz_a_dst[0], sz_a_dst[1], s_sz[0], s_sz[1], chn])
    return lf_up


def spatial_resize(lf, sz_s_dst):
    """
    Resize spatially.
    :param lf: LF matrix (a, a, s1, s1, chn)
    :param sz_s_dst: Destination spatial size (s2, s2)
    :return: Resized LF matrix (a, a, s2, s2, chn)
    """
    a_sz = (lf.shape[0], lf.shape[1])
    s_sz = (lf.shape[2], lf.shape[3])
    chn = lf.shape[4]
    lf = lf.reshape([a_sz[0] * a_sz[1], s_sz[0], s_sz[1], chn])

    lf_up = []
    for i in range(lf.shape[0]):
        up = cv2.resize(lf[i], dsize=(sz_s_dst[1], sz_s_dst[0]), interpolation=cv2.INTER_CUBIC)
        lf_up.append(up)
    lf_up = np.array(lf_up)
    lf_up = lf_up.reshape([a_sz[0], a_sz[1], sz_s_dst[0], sz_s_dst[1], chn])
    return lf_up


def out_spatial_resize(lf, sz_s_dst):
    """
    Resize spatially of output.
    :param lf: LF matrix (a, s1, s1, chn)
    :param sz_s_dst: Destination spatial size (s2, s2)
    :return: Resized LF matrix (a, s2, s2, chn)
    """

    lf_up = []
    for i in range(lf.shape[0]):
        up = cv2.resize(lf[i], dsize=(sz_s_dst[1], sz_s_dst[0]), interpolation=cv2.INTER_CUBIC)
        lf_up.append(up)
    lf_up = np.array(lf_up)
    return lf_up


def raw_preprocess(raw_lf, a_n, a_preserve):
    """A convenient function, not implementation."""
    bgr_lf = lf_raw2bgr(raw_lf, a_n, a_preserve)
    ycrcb_lf = im2double(lf_bgr2ycrcb(bgr_lf))
    y_lf = ycrcb_lf[:, :, :, :, 0]
    return bgr_lf, ycrcb_lf, y_lf


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def path2img_name(pth):
    """Extract img's name from its path."""
    return path.splitext(path.basename(pth))[0]


def transpose_out(ycrcb_up, a_in, a_out):
    """
    Transpose output for Henry's evaluation (in MATLAB).
    For example, ycrcb_up is 60 views in 8*8 output, after transposing it will fit in transposed 8*8 output too.
    """
    in_sai, out_sai = sai_io_idx(a_out, a_in)

    idx_t = np.zeros(a_out[0] * a_out[1]).astype(np.int)
    idx_t[out_sai] = range(len(out_sai))
    idx_t = idx_t.reshape(a_out).transpose((1, 0)).reshape(a_out[0] * a_out[1])
    idx_t = idx_t[out_sai]

    return ycrcb_up[idx_t]
