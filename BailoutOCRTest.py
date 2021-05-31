import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt


def processBailoutImage(oriImg):
    isShowImg = True
    resize_num = 5
    rows, cols = oriImg.shape[0], oriImg.shape[1]
    threshold = 162
    kernel = (3, 3)
    # 平移-1y的矩陣
    M = np.float32([[1, 0, 0], [0, 1, -1]])
    erodeIterationTimes = 2
    dilateIterationTimes = 2

    # First layer Mask:remove noisy line, and extract the trunk of character
    # gray
    grayImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)
    # binary
    retval, binImg = cv2.threshold(grayImg, threshold, 255, cv2.THRESH_BINARY_INV)
    # erode 卷積完會往下跑1pixel，需平移-1y
    erodeImg = cv2.erode(binImg, kernel, iterations=erodeIterationTimes, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    erodeImg = cv2.warpAffine(erodeImg, M, (cols, rows))
    # dilate 卷積完會往下跑1pixel，需平移-1y
    dilateImg = cv2.dilate(erodeImg, kernel, iterations=dilateIterationTimes, borderType=cv2.BORDER_CONSTANT,
                           borderValue=0)
    dilateImg = cv2.warpAffine(dilateImg, M, (cols, rows))
    firstMaskImg = dilateImg
    # 還原影像
    maskImg = cv2.bitwise_and(oriImg, oriImg, mask=dilateImg)
    if isShowImg:
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(binImg, cv2.COLOR_GRAY2RGB))
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(erodeImg, cv2.COLOR_GRAY2RGB))
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(dilateImg, cv2.COLOR_GRAY2RGB))
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(maskImg, cv2.COLOR_BGR2RGB))
        plt.show()

    # Second layer Mask:Keep blue & purple
    # Convert BGR to HSV
    hsv = cv2.cvtColor(oriImg, cv2.COLOR_BGR2HSV)
    # define range of blue & purple color in HSV
    lower_blue = np.array([110, 65, 50])
    upper_blue = np.array([130, 255, 255])
    lower_purple = np.array([125, 50, 46])
    upper_purple = np.array([155, 255, 255])
    # Threshold the HSV image to get only blue & purple colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    secondMaskImg = mask_blue + mask_purple
    # 還原顏色
    maskBlue = cv2.bitwise_and(oriImg, oriImg, mask=secondMaskImg)
    # 反相給tesserate辨識
    resultImg = 255 - secondMaskImg

    if isShowImg:
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB))
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2RGB))
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(mask_purple, cv2.COLOR_GRAY2RGB))
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(secondMaskImg, cv2.COLOR_GRAY2RGB))
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(maskBlue, cv2.COLOR_BGR2RGB))
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB))
        plt.show()

    finalMaskImg = firstMaskImg + secondMaskImg

    # resize
    finalMaskImg = cv2.resize(finalMaskImg,
                              (cols * int(resize_num), rows * int(resize_num)),
                              interpolation=cv2.INTER_CUBIC)
    # 反相給tesserate辨識
    resultImg = 255 - finalMaskImg
    if isShowImg:
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(finalMaskImg, cv2.COLOR_GRAY2RGB))
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB))
        plt.show()
        resultChar = pytesseract.image_to_string(resultImg, lang='eng',
                                                 config='--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789')
    print(resultChar)


if __name__ == "__main__":
    # path = '/Users/moriakiraakira/Downloads/pepper.jpeg'
    path = '/Users/moriakiraakira/Desktop/OCR_Sample/BailoutOCR/30.png'
    oriImg = cv2.imread(path)
    processBailoutImage(oriImg)

    # 10/30 fail
