# -*-coding: utf-8-*-
# author: hxy
"""
使用tensorrt进行视频流的inference
"""

import cv2
import time
import logging
from PIL import Image
import numpy as np
import tensorrt as trt
from lib.trt_inference import InferenceYolov3tiny

TRT_LOGGER = trt.Logger()


# logging set
def log_set():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


# 根据预测的id获取类别名
def get_names(names_file, classes_id):
    with open(names_file, 'r') as f:
        name = f.read()
    names = name.splitlines()
    f.close()
    return names[classes_id]


# 加载model
def load_model(trt_file_path):
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        print("Loading engine: {}!".format(trt_file_path.split('/')[-1]))
        return runtime.deserialize_cuda_engine(f.read())


# 获取视频流并进行推理
def video_inference(rtsp, engine, context):
    logging.info("获取视频流rtsp地址:{}".format(rtsp))
    cap = cv2.VideoCapture('test.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('result_trt_inference.avi', fourcc, 10.0, size)
    num = 0
    while 1:
        _, frame = cap.read()
        start = time.time()
        frame = Image.fromarray(frame, mode="RGB")
        image_raw, boxes, classes_id, scores, ts = engine.preprocess_img(img=frame, context=context)
        num += 1
        print(ts)
        fps = 1 / ts
        img = cv2.cvtColor(np.asarray(image_raw), cv2.COLOR_RGB2BGR)
        for id, box, scores in zip(classes_id, boxes, scores):
            name = get_names(names_file='names.txt',
                             classes_id=id)
            logging.info("{}:{}%".format(name, int(scores * 100)))
            x_coord, y_coord, width, height = box
            left = max(0, np.floor(x_coord + 0.5).astype(int))
            top = max(0, np.floor(y_coord + 0.5).astype(int))
            right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
            bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))
            cv2.rectangle(img, (left - 4, top - 4), (right + 4, bottom + 4), (255, 0, 255), 1)
            cv2.putText(img, name, (left - 5, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img, str('FPS: {%.3f}' % fps), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
            img = img[:, :, [2, 1, 0]]
            out.write(img)
        img = cv2.resize(img, None, fx=.5, fy=.6)
        # img = img[:, :, [2, 1, 0]]
        cv2.imshow("Inference-Results", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    log_set()
    with load_model('yolov3-tiny.trt') as engine, engine.create_execution_context() as context:
        inference_engine = InferenceYolov3tiny(engine=engine)
        # inference_engine = InferenceYolov3(engine=engine)
        video_inference(rtsp=" ",
                        engine=inference_engine,
                        context=context)

