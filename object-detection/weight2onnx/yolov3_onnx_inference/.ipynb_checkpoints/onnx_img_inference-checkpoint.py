# coding: utf-8
# author: hxy
# 2019-12-10

"""
照片的inference；
默认推理过程在CPU上；
"""
import os
import time
import logging
import onnxruntime
from lib.darknet_api import process_img, get_boxes, draw_box


# 定义日志格式
def log_set():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 加载onnx模型
def load_model(onnx_model):
    sess = onnxruntime.InferenceSession(onnx_model)
    in_name = [input.name for input in sess.get_inputs()][0]
    out_name = [output.name for output in sess.get_outputs()]
    logging.info("输入的name:{}, 输出的name:{}".format(in_name, out_name))

    return sess, in_name, out_name


if __name__ == '__main__':
    log_set()
    input_shape = (608, 608)

    # anchors
    anchors_yolo = [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)]]
    anchors_yolo_tiny = [[(81, 82), (135, 169), (344, 319)], [(10, 14), (23, 27), (37, 58)]]
    session, inname, outname = load_model(onnx_model='yolov3_608.onnx')
    logging.info("开始Inference....")
    # 照片的批量inference
    img_files_path = 'test_pic'
    imgs = os.listdir(img_files_path)

    logging.debug(imgs)
    for img_name in imgs:
        img_full_path = os.path.join(img_files_path, img_name)
        logging.debug(img_full_path)
        img, img_shape, testdata = process_img(img_path=img_full_path,
                                               input_shape=input_shape)
        s = time.time()
        prediction = session.run(outname, {inname: testdata})

        # logging.info("推理照片 %s 耗时：% .2fms" % (img_name, ((time.time() - s)*1000)))
        boxes = get_boxes(prediction=prediction,
                          anchors=anchors_yolo,
                          img_shape=input_shape)
        draw_box(boxes=boxes,
                 img=img,
                 img_shape=img_shape)
        logging.info("推理照片 %s 耗时：% .2fms" % (img_name, ((time.time() - s)*1000)))
