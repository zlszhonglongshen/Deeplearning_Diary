import os
import csv
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

if __name__ == '__main__':
    VOCRoot = 'VOC2007'
    csv_file = 'train_labels.csv' # 原始的标注文件
    csv_reader = open(csv_file, 'r', newline='')
    label = {}  # 里面包含了图片的名字和对应的物体位置
    for line, row in enumerate(csv_reader):
        row = row.rstrip().split(',')
        # 去掉非标注行-第一行
        if len(row) != 2:
            continue
        # 得到图片名和对应的物体标注
        img_file, position = row[0], row[1].split()

        if img_file not in label:
            label[img_file] = []

        xmin = position[0]
        ymin = position[1]
        xmax = position[2]
        ymax = position[3]

        # Check that the bounding box is valid.
        if xmax <= xmin or ymax <= ymin:
           print(img_file)

        label[img_file].append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    for img_file in label:
        image_path = VOCRoot + '/JPEGImages/' + img_file
        img = cv2.imread(image_path)
        height, width, channel = img.shape

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'JPEGImages'
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = image_path
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = '%s' % width
        node_height = SubElement(node_size, 'height')
        node_height.text = '%s' % height
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '%s' % channel

        for obj in label[img_file]:
            class_name = 'steel' #
            xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'],  obj['xmax'],  obj['ymax']

            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = class_name
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = '%s' % xmin
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = '%s' % ymin
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = '%s' % xmax
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = '%s' % ymax
            node_name = SubElement(node_object, 'pose')
            node_name.text = 'Unspecified'
            node_name = SubElement(node_object, 'truncated')
            node_name.text = '0'
        xml = tostring(node_root, pretty_print=True) # 'annotation'
        dom = parseString(xml)
        # save_dir = 'VOC2007/Annotations'
        xml_name =  img_file.replace('.jpg', '.xml')
        xml_path = VOCRoot + '/Annotations/' + xml_name
        with open(xml_path, 'wb') as f:
            f.write(xml)
