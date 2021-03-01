import numpy as np
import cv2
import time

start = time.time()
img = cv2.imread(r'img/battery_test_01.jpg')
mask = cv2.imread(r'img/battery_test_01_mask.jpg')
img = img*mask
end = time.time()
print(end-start)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()


# start = time.time()
# rect = (300, 300, 1055, 410)  # 前景区域
#
# # 读取图像
# img = cv2.imread(r'img/battery_test_02.jpg')
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
#
# # 前景提取
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# img = img * mask2[:, :, np.newaxis]
# end = time.time()
# print(end-start)
# # 显示图像
# cv2.imshow("mask", mask)
# cv2.imshow("mask2", mask2)
# cv2.imshow("img", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# # 定义全局变量
# n = 0  # 定义鼠标按下的次数
# ix = 0  # x,y 坐标的临时存储
# iy = 0
# rect = (0, 0, 0, 0)  # 前景区域
#
#
# # 鼠标回调函数
# def draw_rectangle(event, x, y, flags, param):
#     global n, ix, iy, rect
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if n == 0:  # 首次按下保存坐标值
#             n += 1
#             ix, iy = x, y
#             print(x, y)
#         else:  # 第二次按下显示矩形
#             n += 1
#             print(x, y)
#             rect = (ix, iy, (x - ix), (y - iy))  # 前景区域
#
#
# # 读取图像
# img = cv2.imread(r'img/battery_test_01.jpg')
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# # 选择区域 左上到右下矩形
# cv2.namedWindow("img")
# cv2.setMouseCallback("img", draw_rectangle)  # 绑定鼠标
# while (n != 2):
#     cv2.imshow("img", img)
#     cv2.waitKey(2)
# # 前景提取
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# img = img * mask2[:, :, np.newaxis]
# # 显示图像
# cv2.imshow("img", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
