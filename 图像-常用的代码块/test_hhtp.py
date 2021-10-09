# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 18:17
# @Author  : Johnson
import base64
import requests
import json
import time

url = ""
header = {'content-type':'application/json'}

def paddleocr(path):
    image = path
    with open(image,'rb') as f:
        image_70 = base64.b64encode(f.read())
        image_70 = image_70.decode()
    data = {"image_id":12345,"image":image_70}
    response = requests.post(url=url,headers=header,json=data)
    return requests.text


if __name__ == '__main__':
    print(paddleocr("test.jpg"))