import cv2
import numpy as np


def access_pixels(img):
    print(img.shape)  # 获取图片宽高
    height = img.shape[0]
    width = img.shape[1]
    channels = 0
    if len(img.shape) == 3:
        channels = img.shape[2]  # 通道
    print("height: %s , width: %s , channels: %s" % (height, width, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = img[row, col, c]
                img[row, col, c] = 255 - pv
    cv2.imshow("11", img)


def video_demo():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("video", frame)
        c = cv2.waitKey(50)
        if c == 27:
            break


# 效率很高
def inverse_img(img):
    dst = cv2.bitwise_not(img)
    cv2.imshow("d", dst)


# video_demo()

src = cv2.imread("./sample/1.jpg")

# 转混度图像
gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
cv2.imshow("demo1", src)

t1 = cv2.getTickCount()
# access_pixels(src)
inverse_img(src)
t2 = cv2.getTickCount()

print("tick: %s ms" % ((t2 - t1) / cv2.getTickFrequency()))

# cv2.imshow("demo2", gray)
# access_pixels(gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
