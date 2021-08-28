from os import path
import cv2

from components.utils import transpose_out, cvt_ycrcb2bgr, double2im
from components.utils.evaluate import y_upsample


def render_bgr(y_chn, ycrcb, config, pth_dir, transpose=False):
    ycrcb_up, _ = y_upsample(y_chn, ycrcb, config.a_in, config.a_out)
    if transpose:
        ycrcb_up = transpose_out(ycrcb_up, config.a_in, config.a_out)
    for idx_a in range(ycrcb_up.shape[0]):
        bgr_up = cvt_ycrcb2bgr(ycrcb_up[idx_a])
        bgr_up = double2im(bgr_up)
        pth = path.join(pth_dir, "%d.png") % (idx_a + 1)
        cv2.imwrite(pth, bgr_up)
