import os
import numpy as np
import cv2
from pycocotools.coco import COCO
import dlib
from PIL import Image

data_dir    = 'data/coco/'
ann_file    = 'annotations/instances_train2017.json'
img_dir     = 'images/train2017'
face_dir    = 'data/characters'
output_dir  = 'data/samples'

detector = dlib.get_frontal_face_detector()

def edge(img):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.merge((gray, gray, gray), img)
    
    kernel = np.ones((3,3),np.uint8)
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


def deleteBG(img_data):
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

    return cutout_img



def checkHuman(img_data):
    ann_ids     = coco.getAnnIds(imgIds=img_data['id'], iscrowd=None)
    anns        = coco.loadAnns(ann_ids)
    
    human_anns  = []

    for ann in anns:
        cat = [ cat for cat in cats if cat['id'] == ann["category_id"] ][0] # このアノテーションのスーパーカテゴリ
        if cat["supercategory"] == "person":
            human_anns.append(ann)
    
    if human_anns != []:
        return human_anns
    else:
        return None


def replaceCharcter(org_img, human_anns, mode="replace", erase=False):
    output_img = org_img.copy()
    faces = detector(output_img, 0)

    for ann in human_anns:
        cat = [ cat for cat in cats if cat['id'] == ann["category_id"] ][0] # このアノテーションのスーパーカテゴリ
        
        if cat["supercategory"] == "person":
            for seg in ann['segmentation']:
                if erase:
                    if type(seg) == list:   # 多人数の場合は別形式で扱われるため
                        poly = np.array(seg).reshape((int(len(seg)/2), 2)).reshape((-1,1,2)).astype(np.int32)
                        
                        # 白埋め
                        cv2.fillPoly(output_img, pts=[poly], color=(255, 255, 255))
                
            if mode == "replace":
                ann_width   = ann['bbox'][2] - ann['bbox'][0]
                ann_height  = ann['bbox'][3] - ann['bbox'][1]
                c_x         = ann['bbox'][0] + ann_width / 2
                c_y         = ann['bbox'][1] + ann_height / 2

            if mode == "face":
                print(faces)
                for face in faces:
                    x1, y1, x2, y2  = face.left(), face.top(), face.right(), face.bottom()

                    face_width  = int( face.width() )
                    face_height = int( face.height() )
                    # 画像の読み込み -1を付けることでアルファチャンネル
                    cface_img = cv2.imread(
                        os.path.join(
                            face_dir, 
                            "man_face.png" ), -1 )

                    # 顔サイズにリサイズ
                    cface_img = cv2.resize( cface_img, ( face_width, face_height ) )
                    
                    # 背景(org_img)をPIL形式に変換
                    src     = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                    pil_src = Image.fromarray(src)
                    pil_src = pil_src.convert('RGBA')

                    # オーバーレイ(cface_img)をPIL形式に変換
                    overlay     = cv2.cvtColor(cface_img, cv2.COLOR_BGRA2RGBA)
                    pil_overlay = Image.fromarray(overlay)
                    pil_overlay = pil_overlay.convert('RGBA')

                    # 画像を合成
                    pil_tmp         = Image.new( 'RGBA', pil_src.size, (255, 255, 255, 0) )
                    pil_tmp.paste( pil_overlay, (x1, y1), pil_overlay )
                    result_image    = Image.alpha_composite(pil_src, pil_tmp)

                    # OpenCV形式に変換
                    output_img  = cv2.cvtColor( np.asarray( result_image ), cv2.COLOR_RGBA2BGRA )
                    output_img  = cv2.cvtColor( output_img, cv2.COLOR_BGRA2BGR )

    return output_img


        


if __name__ == "__main__":
    coco    = COCO(os.path.join(data_dir, ann_file))
    cats        = coco.loadCats(coco.getCatIds())

    image_files = sorted(os.listdir(os.path.join(data_dir, img_dir)))

    # img_ids     = [int(img_file.split('.')[0]) for img_file in image_files]
    img_ids     = [int(img_file.split('.')[0]) for img_file in image_files][0:3]   # To develoment

    for idx, img_id in enumerate(img_ids):
        # img_data    = coco.loadImgs(img_ids[np.random.randint(0,len(img_ids))])[0]     # To develoment
        img_data    = coco.loadImgs(img_id)[0]
        print(img_data["file_name"])
        # 画像の読み込み
        org_img = cv2.imread(os.path.join(data_dir, img_dir, img_data['file_name']))

        
        if checkHuman( img_data ):
            human_anns   = checkHuman( img_data )
            output_img  = replaceCharcter( org_img, human_anns, mode="face" )
        else: 
            output_img  = org_img
            print(" No Human. ")
        
        # cutout_img = deleteBG(img_data)
        # output_img = edge(org_img)

        # output_img = binarization(cutout_img)
        # cv2.namedWindow('org')
        # cv2.imshow('org', org_img)
        # cv2.namedWindow('mask')
        # cv2.imshow('mask', mask_img)

        # cv2.namedWindow('original')
        # cv2.imshow('original', org_img)
        # cv2.imwrite('original.jpg', org_img)

        cv2.namedWindow('output')
        cv2.imshow('output', output_img)

        # cv2.imwrite(os.path.join(output_dir, str(img_id).zfill(12) + ".jpg"), output_img)
        print(str(idx+1).rjust(20), "/", len(img_ids))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        