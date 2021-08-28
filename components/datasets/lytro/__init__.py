import cv2

from components.utils import lf_raw2bgr, im2double, lf_bgr2ycrcb


def load_item(pth_img, a_raw, a_preserve):
    """
    Load structured BGR and YCrCb matrix from path of a raw LF.
    :param pth_img: Path of a raw LF.
    :param a_raw: Original angular size, e.g. [14, 14].
    :param a_preserve: Preserved angular size, e.g. [8, 8].
    :return: Structured BGR and YCrCb matrix.
    """
    raw_lf = cv2.imread(pth_img, cv2.IMREAD_UNCHANGED)
    bgr_lf, ycrcb_lf = raw_preprocess(raw_lf, a_raw, a_preserve)

    return bgr_lf, ycrcb_lf


def raw_preprocess(raw_lf, a_raw, a_preserve):
    """
    Process raw LF into structured BGR and YCrCb matrix.
    The center part with a_preserve size within the a_raw angular will be preserved.
    :param raw_lf: Raw LF image.
    :param a_raw: Original angular size, e.g. [14, 14].
    :param a_preserve: Preserved angular size, e.g. [8, 8].
    :return: Structured BGR and YCrCb matrix.
    """
    bgr_lf = lf_raw2bgr(raw_lf, a_raw, a_preserve)
    ycrcb_lf = im2double(lf_bgr2ycrcb(bgr_lf))
    return bgr_lf, ycrcb_lf
