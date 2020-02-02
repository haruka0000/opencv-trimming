from PIL import Image
import cv2
import numpy as np

###########################
# エッジ抽出
###########################
def edge(img, ksize=3):
    img = img.copy()

    gray = img
    if len(img.shape) == 3:
        # グレースケール変換
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.merge((gray, gray, gray), img)
    
    kernel = np.ones((ksize,ksize),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 1)
    
    diff = cv2.subtract(dilation, img)
    
    negaposi = 255 - diff

    return negaposi


def laplacian(img, ksize=3):
    if len(img.shape) == 3:
        # グレースケール変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img, cv2.CV_32F, ksize=ksize)

    
###########################
# 二値化
###########################
def binarization(img, threshold=120):
    # 二値化(閾値100を超えた画素を255にする。)
    ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img_thresh



###########################
# DoG フィルタ
###########################
# ガウシアン差分フィルタリング
def DoG(img, ksize=3, sigma=1, k=1.6, gamma=1):
    img = img.copy()
    gray = img
    if len(img.shape) == 3:
        # グレースケール変換
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    g2 = cv2.GaussianBlur(gray, (ksize, ksize), sigma*k)
    return g1 - gamma*g2

# 閾値で白黒化するDoG
def thres_dog(img, size, sigma, eps, k=1.6, gamma=0.98):
    d = DoG(img,size, sigma, k, gamma)
    d /= d.max()
    d *= 255
    d = np.where(d >= eps, 255, 0)
    return d


###########################
# XDoG フィルタ
###########################
# 拡張ガウシアン差分フィルタリング
def xDoG(img, size, sigma, eps, phi, k=1.6, gamma=0.98):
    if len(img.shape) == 3:
        # グレースケール変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eps /= 255
    d = DoG(img,size, sigma, k, gamma)
    d = d / d.max()
    e = 1 + np.tanh( phi * (d-eps) )
    e[e>=1] = 1
    return e



###########################
# p_XDoG フィルタ (p: シャープネスパラメータ)
###########################
# シャープネス値pを使う方
def p_xDoG(img, size, p, sigma, eps, phi, k=1.6):
    if len(img.shape) == 3:
        # グレースケール変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eps /= 255
    g1 = cv2.GaussianBlur(img, (size, size), sigma)
    g2 = cv2.GaussianBlur(img, (size, size), sigma*k)
    d = (1 + p) * g1 - p * g2
    d = d / d.max()
    e = 1 + np.tanh(phi*(d-eps))
    e[e>=1] = 1
    return e
