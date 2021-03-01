import cv2

img = cv2.imread(r"img/a1.jpeg")

cv2.imwrite(r"img/a01.jpeg", img[0:433, 0:316])
cv2.imwrite(r"img/a02.jpeg", img[0:433, 324:640])
