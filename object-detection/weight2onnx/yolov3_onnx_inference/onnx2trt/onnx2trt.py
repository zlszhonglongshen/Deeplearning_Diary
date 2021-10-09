# -*-coding: utf-8-*-
# author: HXY
# 2019-12-24
"""
tensorrt6.0
https://blog.csdn.net/weixin_38106878/article/details/106364820?spm=1001.2014.3001.5501
"""

import os
import tensorrt as trt

TRT_LOGGER = trt.Logger()


class GetTrt(object):
    def __init__(self, onnx_file_path, trt_save_path):
        self.onnx_file_path = onnx_file_path
        self.batch_size = 1
        self.fp16_on = True
        self.trt_save_path = trt_save_path

    def build_engine(self):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network,
                                                                                                     TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # 30:1GB; 28:256MiB
            builder.max_batch_size = 1
            builder.fp16_mode = self.fp16_on
            if not os.path.exists(self.onnx_file_path):
                print("Onnx file not found")
                exit(0)
            print("loading onnx file from path {}".format(self.onnx_file_path))
            with open(self.onnx_file_path, 'rb') as model:
                parser.parse(model.read())
            print("Completed parsing of onnx file....")
            print("building an engine..this may take a while....")
            # network.get_input(0).shape = [1, 3, 416, 416]
            engine = builder.build_cuda_engine(network)
            with open(self.trt_save_path, 'wb') as f:
                f.write(engine.serialize())
            print("create engine completed")


"""
test function
"""
if __name__ == '__main__':
    test = GetTrt('./yolov3-tiny.onnx',
                  './yolov3-tiny.trt')
    test.build_engine()
