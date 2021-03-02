import cv2
import numpy as np
i = 6
img = cv2.imread(r"img/image{}.jpg".format(i))

# img = img[255:365, 0:1055]
# mask_temp = np.ones(img.shape[:2])
#
# img = cv2.copyMakeBorder(img, 350, 250, 300, 300, cv2.BORDER_CONSTANT, value=[0, 255, 0])
# mask = cv2.copyMakeBorder(mask_temp, 350, 250, 300, 300, cv2.BORDER_CONSTANT, value=0)
#
# cv2.imwrite(r"img/battery_test_02.jpg", img)
# cv2.imwrite(r"img/battery_test_02_mask.jpg", mask)
# cv2.imwrite(r"img/a02.jpeg", img[0:433, 324:640])
print(img.shape[0])
im1 = img[0:int(img.shape[0]/2), 0:int(img.shape[1])]
im2 = img[int(img.shape[0]/2):int(img.shape[0]), 0:int(img.shape[1])]
cv2.imshow('im1', im1)
cv2.imshow('im2', im2)
cv2.imwrite(r"img/image{}-0.png".format(i), im1)
cv2.imwrite(r"img/image{}-1.png".format(i), im2)
cv2.waitKey()
cv2.destroyAllWindows()
