# -*- coding: utf-8 -*-
"""
Created on 2020/7/8 00:03
@author: Johnson
Email:593956670@qq.com
"""
import argparse
import cv2

from yolo import YOLO

print("loading yolo-tiny...")
# yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
# yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])

yolo.size = int(416)
yolo.confidence = float(0.2)

image = cv2.imread("./images/christy-alby-LFeid8CY0Os-unsplash.jpg")
width, height, inference_time, results = yolo.inference(image)

for detection in results:
    id, name, confidence, x, y, w, h = detection
    cx = x + (w / 2)
    cy = y + (h / 2)

    # draw a bounding box rectangle and label on the image
    color = (0, 255, 255)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = "%s (%s)" % (name, round(confidence, 2))
    print(text)
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

cv2.imshow("preview", image)
key = cv2.waitKey(0)
cv2.destroyWindow("preview")
