import cv2
import numpy as np
from skimage.metrics import structural_similarity

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def registerImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography
    height, width, channels = im1.shape
    imgReg = cv2.warpPerspective(im2, h, (width, height))

    return imgReg, h


def findDifferentImages(im1, im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("im1Gray", im1Gray)
    # cv2.imshow("im2Gray", im2Gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # 计算两个灰度图像之间的结构相似度指数
    (score, diff) = structural_similarity(im1Gray, im2Gray, full=True)
    # (score, diff) = structural_similarity(im1, im2, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM:{}".format(score))
    # diff = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # diff = cv2.subtract(im1, im2)
    # cv2.imshow('diff', diff)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # 找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # 开操作。扩大图像缺口
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
    binary = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se)

    # cv2.imshow('diff', diff)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow("binary", binary)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = im2.copy()
    # 找到一系列区域，在区域周围放置矩形
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # result = cv2.drawContours(im2.copy(), contours, -1, (0, 0, 0), 1)
    # 用cv2.imshow 展现最终对比之后的图片， cv2.imwrite 保存最终的结果图片
    return result
