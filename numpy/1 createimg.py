import cv2 as cv
import numpy as np


def create_single_img():
    img = np.zeros([400, 400, 1], np.uint8)
    img[:, :, 0] = np.ones([400, 400]) * 255
    cv.imshow("demo", img)


# 创建3通道
def create_img():
    """
    :return:
    """
    img = np.zeros([400, 400, 3], np.uint8)
    img[:, :, 0] = np.ones([400, 400]) * 255
    cv.imshow("demo", img)


def createArr():
    m1 = np.ones([3, 3, 3], np.uint8)
    m1.fill(111)
    print(m1)


create_single_img()
createArr()

cv.waitKey(0)
cv.destroyAllWindows()
