import numpy as np
import cv2 as cv

img = cv.imread(r'img/battery_test.jpg', 1)
rows, cols, channels = img.shape
p1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [rows-1,cols-1]])
p2 = np.float32([[0,rows*0.3], [cols*0.8,rows*0.2], [cols*0.15,rows*0.7], [cols*0.8,rows*0.8]])
M = cv.getPerspectiveTransform(p1,p2)
dst = cv.warpPerspective(img, M, (cols, rows))
cv.imshow('original', img)
cv.imshow('result', dst)
cv.waitKey(0)
cv.destroyAllWindows()