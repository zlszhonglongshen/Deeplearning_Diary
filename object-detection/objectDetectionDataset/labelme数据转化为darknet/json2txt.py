# -*- coding:utf-8 -*-

from __future__ import print_function
import os, sys, zipfile
import numpy as np
import json

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    #x = (box[0] + box[1])/2.0 - 1
    #y = (box[2] + box[3])/2.0 - 1
    #w = box[1] - box[0]
    #h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


json_file='d:/train.json' # # Object Instance 类型的标注


data=json.load(open(json_file,'r'))
#print(data["images"])

ana_txt_save_path = "d:/val_coco2017/"
if not os.path.exists(ana_txt_save_path):
    os.makedirs(ana_txt_save_path)

for img in data['images']:
    print(img["id"])
    #print(img["file_name"])
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    #print(img["height"])
    #print(img["width"])
    img_id = img["id"]
    ana_txt_name = filename.split(".")[0] + ".txt"
    f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
    for ann in data['annotations']:
        if ann['image_id']==img_id:
            #annotation.append(ann)
            #print(ann["category_id"], ann["bbox"])
            box = convert((img_width,img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n"%(ann["category_id"], box[0], box[1], box[2], box[3]))
    f_txt.close()