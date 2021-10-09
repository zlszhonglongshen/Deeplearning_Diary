# -*- coding: utf-8 -*-
"""
Created on 2020/4/18 10:06
@author: Johnson
Email:593956670@qq.com
"""


path = "C:/Users/Johnson/Desktop/test/labelme_json/"

import os

dirs = os.listdir(path)
for file_name in dirs:
    # print(file_name)
    src = (os.path.join(path+file_name,"label.png"))
    num = file_name.split("_json")[0]
    # print(num)
    os.chdir(r"C:\Users\Johnson\Desktop\test\cv2_mask")
    # dst = os.path.join("C:/Users/Administrator/Desktop/test/cv2_mask/","{}.png".format(num))
    os.rename(src, "{}.png".format(num))