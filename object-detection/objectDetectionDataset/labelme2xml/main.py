# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2020/06/10 17:55
@Author  : Johnson
@Email   : 593956670@qq.com
"""
import os
from tqdm import tqdm

from read_json_anno import ReadAnno
from create_xml_anno import CreateAnno


def json_transform_xml(json_path, xml_path, process_mode="rectangle"):
    json_path = json_path
    json_anno = ReadAnno(json_path, process_mode=process_mode)
    width, height = json_anno.get_width_height()
    filename = json_anno.get_filename()
    coordis = json_anno.get_coordis()

    xml_anno = CreateAnno()
    xml_anno.add_filename(filename)
    xml_anno.add_pic_size(width_text_str=str(width), height_text_str=str(height), depth_text_str=str(3))
    for xmin, ymin, xmax, ymax, label in coordis:
        xml_anno.add_object(name_text_str=str(label),
                            xmin_text_str=str(int(xmin)),
                            ymin_text_str=str(int(ymin)),
                            xmax_text_str=str(int(xmax)),
                            ymax_text_str=str(int(ymax)))

    xml_anno.save_doc(xml_path)


if __name__ == "__main__":
    root_json_dir = r"C:/Users/Johnson/Desktop/pic/json"
    root_save_xml_dir = r"C:/Users/Johnson/Desktop/cc/"
    for json_filename in tqdm(os.listdir(root_json_dir)):
        try:
            json_path = os.path.join(root_json_dir, json_filename)
            save_xml_path = os.path.join(root_save_xml_dir, json_filename.replace(".json", ".xml"))
            json_transform_xml(json_path, save_xml_path, process_mode="polygon")
        except :
            print(json_path)