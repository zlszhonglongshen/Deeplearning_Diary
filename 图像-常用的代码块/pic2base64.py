# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 17:59
# @Author  : Johnson
import base64

txt = open("test.json","a")
with open("test.jpg","rb") as f:
    base_data = base64.b64encode(f.read())
    txt.write(base_data.decode())
txt.close()

