import cv2

img = cv2.imread(r"img/battery.jpg")

img = img[255:365, 0:1055]

img = cv2.copyMakeBorder(img, 300, 300, 300, 300, cv2.BORDER_CONSTANT, value=[0, 0, 0])

cv2.imwrite(r"img/battery_test.jpg", img)
# cv2.imwrite(r"img/a02.jpeg", img[0:433, 324:640])
