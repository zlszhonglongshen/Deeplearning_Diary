# -*-coding: utf-8-*-
# author: hxy
"""
Inference:
yolov3-tiny.trt
"""

import time
import cv2
from lib import common
import tensorrt as trt
from lib.data_processing import PreprocessYOLO, PostprocessYOLO, VideoPreprocessYOLO

TRT_LOGGER = trt.Logger()


# def load_model(trt_file_path):
#     with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
#         print("Loading engine from: {}".format(trt_file_path.split('/')[-1]))
#         return runtime.deserialize_cuda_engine(f.read())

# Inference on Yolov3-tiny
class InferenceYolov3tiny(object):
    def __init__(self, engine):
        self.engine = engine
        # self.img_path = test_img_path
        self.input_size = (416, 416)
        self.postprocess_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],
                                 "yolo_anchors": [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)],
                                 "obj_threshold": 0.5,
                                 "nms_threshold": 0.35,
                                 "yolo_input_resolution": (416, 416)}
        self.output_shape_416 = [(1, 24, 13, 13), (1, 24, 26, 26)]
        self.output_shape_480 = [(1, 24, 15, 15), (1, 24, 30, 30)]
        self.output_shape_544 = [(1, 24, 17, 17), (1, 24, 34, 34)]
        self.output_shape_608 = [(1, 24, 19, 19), (1, 24, 38, 38)]
        print("Image input size:{}".format(self.input_size))

    def preprocess_img(self, img, context):
        # preprocessor = PreprocessYOLO(self.input_size) # 照片处理函数
        preprocessor = VideoPreprocessYOLO(self.input_size)  # 视频流处理函数
        image_raw, image = preprocessor.process(img)
        ori_img_hw = image_raw.size
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
        inputs[0].host = image
        s = time.time()
        trt_outputs = common.do_inference(context=context,
                                          bindings=bindings,
                                          inputs=inputs,
                                          outputs=outputs,
                                          stream=stream)
        ts = time.time() - s
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shape_416)]
        postprocessor = PostprocessYOLO(**self.postprocess_args)
        boxes, classes, scores = postprocessor.process(trt_outputs, ori_img_hw)
        print("Inferences cost: %.3f ms" % ((time.time() - s) * 1000))
        if boxes is None:
            return image_raw, [], [], [], ts
        else:
            return image_raw, boxes, classes, scores, ts

