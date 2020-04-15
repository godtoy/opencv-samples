import os

import cv2 as cv
import numpy as np


def match_tpl(tpl, target):
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        res = cv.matchTemplate(target, tpl, md)
        # opencv 的函数 minMaxLoc：在给定的矩阵中寻找最大和最小值，并给出它们的位置
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
            # 因为滑块只需要 x 坐标的距离，放回坐标元组的 [0] 即可
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.imshow("match-" + np.str(md), target)


# 查找轮廓
def edge_demo(f, img):
    dst = cv.GaussianBlur(img, (3, 3), 0)  # 高斯模糊
    dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    cv.imshow(f, dst)
    xgrad = cv.Sobel(dst, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(dst, cv.CV_16SC1, 0, 1)
    do = cv.Canny(xgrad, ygrad, 30, 350)
    return do


# 处理
def handle_all_path(p):
    files = os.listdir(p)
    for i, f in enumerate(files):
        if i > 0:
            break
        f = os.path.join(p, f)
        src = cv.imread(f)
        w = 344
        # 第一部分
        panel = cv.bitwise_not(src[:w, :])
        cv.imshow(f, panel)

        # 第二部分
        codes = src[w:w + 40, :]
        codes = cv.bitwise_not(codes)
        cv.imshow("aaa", codes)
        # match_tpl(a, panel)

        # 对物体进行降噪
        # codes = edge_demo(f, codes)
        # contours_demo(codes)
        # contours_demo(panel)

        # _, dst = cv.threshold(dst, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)  # 将灰度图像转成二值图像
        # dst = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10) # 自适应
        # dst = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10) # 高斯二值化
        # cv.imshow(f, dst)
        # contours_demo(dst)


def contours_demo(img):
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    print(len(contours))
    cv.drawContours(img, contours, -1, (0, 0, 255), 2)
    cv.imshow("val", img)


handle_all_path("./sample")
cv.waitKey(0)
cv.destroyAllWindows()
