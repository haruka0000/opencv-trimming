import os
import numpy as np
import cv2
from pycocotools.coco import COCO

data_dir    = 'data'
ann_file    = 'annotations_trainval2017/annotations/instances_val2017.json'
img_dir     = 'val2017/'



def edge(img):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.merge((gray, gray, gray), img)
    
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 1)
    
    diff = cv2.subtract(dilation, img)
    
    negaposi = 255 - diff

    return negaposi


def binarization(img):
    # 閾値の設定
    threshold = 150
    # 二値化(閾値100を超えた画素を255にする。)
    ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img_thresh
    

if __name__ == "__main__":
    coco    = COCO(os.path.join(data_dir, ann_file))
    image_files = sorted(os.listdir(os.path.join(data_dir, img_dir)))


    img_ids     = [int(img_file.split('.')[0]) for img_file in image_files]
    img_data    = coco.loadImgs(img_ids[np.random.randint(0,len(img_ids))])[0]
    ann_ids     = coco.getAnnIds(imgIds=img_data['id'], iscrowd=None)
    anns        = coco.loadAnns(ann_ids)

    # 画像の読み込み
    org_img = cv2.imread(os.path.join(data_dir, img_dir, img_data['file_name']))
    # 画像の大きさを取得
    height, width, channels = org_img.shape[:3]

    #ブランク画像
    mask_img = np.zeros((height, width, 3), np.uint8)

    # polygon
    for ann in anns:
        for seg in ann['segmentation']:
            if type(seg) == list:
                poly = np.array(seg).reshape((int(len(seg)/2), 2)).reshape((-1,1,2)).astype(np.int32)
                # print(poly)
                cv2.fillPoly(mask_img, pts=[poly], color=(255, 255, 255))

    gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    cutout_img = org_img.copy()
    # マスク画像合成
    cutout_img[gray_mask==0] = [255, 255, 255]  # マスク画像の明度 0 の画素を白色（R:255 G:255 B:255）で塗りつぶす

    output_img = edge(cutout_img)
    # output_img = binarization(cutout_img)
    # cv2.namedWindow('org')
    # cv2.imshow('org', org_img)
    # cv2.namedWindow('mask')
    # cv2.imshow('mask', mask_img)
    cv2.namedWindow('cutout')
    cv2.imshow('cutout', cutout_img)
    cv2.namedWindow('output')
    cv2.imshow('output', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()