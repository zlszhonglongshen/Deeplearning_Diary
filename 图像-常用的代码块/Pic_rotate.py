# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 18:01
# @Author  : Johnson
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from imutils import paths


def rotate_image(path,angle):
    img = Image.open(path)
    img = np.array(img)
    if angle == 90:
        im = Image.fromarray(img).transpose(Image.ROTATE_90).convert("RGB")
    elif angle == 180:
        im = Image.fromarray(img).transpose(Image.ROTATE_180).convert("RGB")
    elif angle == 270:
        im = Image.fromarray(img).transpose(Image.ROTATE_270).convert("RGB")
    elif angle == 0:
        im = Image.fromarray(img)
    im = im.convert('RGB')
    return im.save(path)

for path in tqdm(paths.list_images(("data/90"))):
    rotate_image(path,90)