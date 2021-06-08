import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import time
import os


# https://guar.smeg.org.tw/Login.aspx

def getVProjection(image):
    vProjection = np.zeros(image.shape, np.uint8);
    (h, w) = image.shape
    # print('img w:', w, ',h:', h)
    # 長度與圖寬一致的組
    w_ = [0] * w

    # 循環統計每一列白色像素的個數
    for x in range(w):
        for y in range(h):
            # 本支流程若使用255會出事
            if image[y, x] > 170:
                w_[x] += 1

    # 繪製垂直平投影圖像
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255

    #cv2.imshow('vProjection',vProjection)
    # plt.imshow(cv2.cvtColor(vProjection, cv2.COLOR_GRAY2RGB))
    # plt.show()
    return w_


def splitAlphabet(projectImg, oriImg, resultImg):
    position = []
    blackPixelThreshold = 80
    W = getVProjection(projectImg)
    H_Start = 0
    H_End = np.size(projectImg, 0)
    Wstart = 0
    Wend = 0
    W_Start = 0
    W_End = 0
    for j in range(len(W)):
        if W[j] > 0 and Wstart == 0:
            W_Start = j
            Wstart = 1
            Wend = 0
        if W[j] <= 0 and Wstart == 1:
            W_End = j
            Wstart = 0
            Wend = 1
        if Wend == 1:
            position.append([W_Start, H_Start, W_End, H_End])
            Wend = 0
    splitImg = []
    splitImgW = 70  # 切割後固定每張圖的寬度
    random_num = 0  # 變數
    for m in range(len(position)):
        if oriImg is not None:
            cv2.rectangle(oriImg, (position[m][0], position[m][1]), (position[m][2], position[m][3]), (0, 255, 255),
                          1)
        tmpH = position[m][3] - position[m][1]
        tmpW = position[m][2] - position[m][0]
        tmpImg = resultImg[position[m][1]:position[m][1] + tmpH, position[m][0]:position[m][0] + tmpW]
        tmpW = tmpImg.shape[1]
        paddingW = int(splitImgW / 2 - tmpW / 2)
        # 切割後等寬
        if tmpW < splitImgW:
            if tmpW % 2 > 0:
                tmpImg = cv2.copyMakeBorder(tmpImg, 0, 0, paddingW + 1, paddingW, cv2.BORDER_CONSTANT,
                                            value=[255, 255, 255])
            else:
                tmpImg = cv2.copyMakeBorder(tmpImg, 0, 0, paddingW, paddingW, cv2.BORDER_CONSTANT,
                                            value=[255, 255, 255])
        # 算黑色pixel數
        #print(np.where(tmpImg == 0)[0].shape[0])
        if np.where(tmpImg == 0)[0].shape[0] < blackPixelThreshold:
            continue
        # -----
        cv2.imwrite('temp/' + str(random_num) + '.png', tmpImg)
        splitImg.append('temp/' + str(random_num) + '.png')
        random_num += 1
    return splitImg


def findPaths(directory):
    # directory = r'/Users/moriakiraakira/Desktop/OCR_Sample/cathayOCR/Source'
    filePaths = []
    for entry in os.scandir(directory):
        if (entry.path.endswith(".jpg")
            or entry.path.endswith(".png")) and entry.is_file():
            print(entry.path)
            filePaths.append(entry.path)
    return filePaths


def processBailoutImage(oriImg):
    isShowImg = False
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
    oriImg = cv2.resize(oriImg, (cols * int(resize_num), rows * int(resize_num)), interpolation=cv2.INTER_CUBIC)
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
    # resultChar = pytesseract.image_to_string(resultImg, lang='eng',
    #                                          config='--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789')

    # 分割
    splitImgs = splitAlphabet(projectImg=finalMaskImg, oriImg=oriImg, resultImg=resultImg)
    # if isShowImg:
    #     for i in range(len(splitImgs)):
    #         plt.subplot(220 + i + 1), plt.imshow(cv2.cvtColor(splitImgs[i], cv2.COLOR_GRAY2RGB))
    #     plt.show()
    return splitImgs

# def cathayOCR_Bailout(oriImg):
#     splitImgs = processBailoutImage(oriImg)
#
#     # 產出切割文字檔
#     resultText = []
#     for img in splitImgs:
#         resultChar = pytesseract.image_to_string(img, lang='eng',
#                                                  config='--psm 10 -c tessedit_char_whitelist=abcdefghijkmnopqrstuvwxyz023456789')
#
#         # 存檔--
#         timestamp = int(time.time() * 1e6)
#         filePath = "/Users/moriakiraakira/Desktop/OCR_Sample/BailoutOCR/Char/{resultChar}_{timestamp}.png".format(
#             resultChar=resultChar, timestamp=timestamp)
#         cv2.imwrite(filePath, img)
#         resultText.append(resultChar)
#         # --
#     return resultText


# best[3,150,3,3,7]
def split_image(path):
    oriImg = cv2.imread(path)
    splitImgs = processBailoutImage(oriImg)

    return splitImgs