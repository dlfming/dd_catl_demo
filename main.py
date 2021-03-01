import cv2
import utils
import time

if __name__ == '__main__':
    start = time.time()
    i = 9
    # refFilename = r"img/book01.jpg"
    imReference = cv2.imread(r"img/image0{}-0.png".format(i), cv2.IMREAD_COLOR)

    # Read image to be aligned
    # imFilename = r"img/battery_test_02.jpg"
    im = cv2.imread(r"img/image0{}-1.png".format(i), cv2.IMREAD_COLOR)

    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imRegister, h = utils.registerImages(imReference, im)
    registerTime = time.time()
    print('registerTime:{}'.format(registerTime-start))
    imResult = utils.findDifferentImages(imReference, imRegister)

    findDifferentTime = time.time()
    print('findDifferentTime:{}'.format(findDifferentTime-registerTime))

    cv2.imshow('imReference', imReference)
    cv2.imshow('registered', imRegister)
    cv2.imshow('result', imResult)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Write aligned image to disk.
    # outFilename = "aligned.jpg"
    # cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
    pass
