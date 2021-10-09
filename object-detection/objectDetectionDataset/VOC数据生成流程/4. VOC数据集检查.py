import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
import matplotlib.pyplot as plt
from math import sqrt as sqrt

# 需要检查的数据
sets=[('2007', 'train'), ('2007', 'val')]

# 需要检查的类别
classes = ['steel']

if __name__ == '__main__':
    # GT框宽高统计
    width = []
    height = []

    for year, image_set in sets:
        # 图片ID不带后缀
        image_ids = open('VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        for image_id in image_ids:
            # 图片的路径
            img_path = 'VOC%s/JPEGImages/%s.jpg'%(year, image_id)
            # 这张图片的XML标注路径
            label_file = open('VOC%s/Annotations/%s.xml' % (year, image_id))
            tree = ET.parse(label_file)
            root = tree.getroot()
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            img = cv2.imread(img_path)
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 2:
                    continue
                cls_id = classes.index(cls)

                xmlbox = obj.find('bndbox')
                xmin = int(xmlbox.find('xmin').text)
                ymin = int(xmlbox.find('ymin').text)
                xmax = int(xmlbox.find('xmax').text)
                ymax = int(xmlbox.find('ymax').text)
                w = xmax - xmin
                h = ymax - ymin

                #img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 8)
                w_change = (w / img_w) * 416
                h_change = (h / img_h) * 416
                s = w_change * h_change
                width.append(sqrt(s))
                height.append(w_change / h_change)
            print(img_path)
            # img = cv2.resize(img, (608, 608))
            # cv2.imshow('result', img)
            # cv2.waitKey()

    plt.plot(width, height, 'ro')
    plt.show()