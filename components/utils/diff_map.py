import cv2

from components.utils import im2double


def diff_map(img1, img2):
    diff = (abs((im2double(img1) - im2double(img2))).sum(axis=2) / 3 * 1500).astype("uint8")
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return diff_color
