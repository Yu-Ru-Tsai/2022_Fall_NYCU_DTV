import os
import cv2
import numpy as np
import xmltodict
import torch
from yolox.data import ValTransform

def read_filename_list(path, file):
    with open(os.path.join(path, file+'.txt'), mode='r') as fp:
        filename_list = [filename.rstrip() for filename in fp.readlines()]
    return filename_list

def write_annotation_xml(path, file, bboxes, IMG_H=1080, IMG_W=1920):
    xml_dict = {
        "annotation": {
            "folder": "VOC2022",
            "filename": file + ".jpg",
            "size": {
                "width": IMG_W,
                "height": IMG_H,
                "depth": 3
            },
            "segmented": 1,
            "object": bboxes
        }
    }

    with open(os.path.join(path, file + '.xml'), mode='w') as fp:
        fp.write(xmltodict.unparse(xml_dict, pretty=True))
        
def read_annotation_txt(path, file, IMG_H=1080, IMG_W=1920):
    with open(os.path.join(path, file+'.txt'), mode='r') as fp:
        bboxes = [line.strip().split(' ') for line in fp.readlines()]

    objects = []
    for bbox in bboxes:
        _, x, y, w, h = map(float, bbox)
        objects.append({
            'name': 'car',
            'pose': 'Unspecified',
            'truncated': 0,
            'difficult': 0,
            'bndbox': {
                'xmin': round((x - w / 2) * IMG_W),
                'ymin': round((y - h / 2) * IMG_H),
                'xmax': round((x + w / 2) * IMG_W),
                'ymax': round((y + h / 2) * IMG_H),
            }
        })
    return objects

def padding(img, target_size, swap=(2, 0, 1)):
    # 參考data_augment.py
    assert (not target_size[0] % 32) and (not target_size[1] % 32), "Not divisible by 32"
    padded_img = np.ones(target_size, dtype=np.uint8) * 114

    padded_img[:img.shape[0], :img.shape[1]] = img
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    
    torch_img = torch.from_numpy(padded_img)
    torch_img = torch_img.unsqueeze(0)
    return torch_img


def load_image(path, filename, IMG_H=1080, IMG_W=1920):
    # 要能被32整除
    img_loc = os.path.join(path, filename + '.jpg')
    img = cv2.imread(img_loc, cv2.IMREAD_COLOR)

    IMG_H = IMG_H + (32 - IMG_H % 32) if IMG_H % 32 else IMG_H
    IMG_W = IMG_W + (32 - IMG_W % 32) if IMG_W % 32 else IMG_W

    img = padding(img, target_size=(IMG_H, IMG_W, 3))
    return img

def iou(box1, box2):
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2, _ = box2
    xmin = max(x1 - w1/2, x2 - w2/2)
    xmax = min(x1 + w1/2, x2 + w2/2)
    ymin = max(y1 - h1/2, y2 - h2/2)
    ymax = min(y1 + h1/2, y2 + h2/2)

    int_area = (xmax - xmin) * (ymax - ymin)
    uni_area = (w1 * h1) + (w2 * h2) - int_area

    return int_area / uni_area

def nms(bboxs, threshold):
    selected = []
    bboxs.sort(key=lambda bbox : bbox[0])

    while len(bboxs):
        selected.append(bboxs.pop())
        selected_bbox = selected[-1]
        bboxs = [bbox for bbox in bboxs if iou(selected_bbox, bbox) < threshold]

    return selected

def Write_Inference_txt(path, file, bboxes, IMG_H=1080, IMG_W=1920):
    with open(os.path.join(path, file + '.txt'), mode='w') as fp:
        for x, y, w, h, conf in bboxes:
            x /= IMG_W
            w /= IMG_W
            y /= IMG_H
            h /= IMG_H
            fp.write(f"0 {conf} {x} {y} {w} {h}\n")
    