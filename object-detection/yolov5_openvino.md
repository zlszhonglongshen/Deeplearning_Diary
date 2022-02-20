# yolov5-V2版本
此次转换都是基于yoloV2版本。
# openvino安装
## 环境变量配置
[参考文章](https://www.jianshu.com/p/4a15bdab767c)
# python环境依赖
```
conda activate openvino  # 进入ubuntu 的虚拟环境
git clone https://github.com/ultralytics/yolov5.git
cd yolov5pip3 install -r requirements.txt onnx# 降版本
pip install torch==1.5.1 torchvision==0.6.1
```
# 模型训练以及模型导出
导出训练好的yoloV5模型，并放入到特定目录下


# 要下载 v2.0 with nn.LeakyReLU(0.1) 的版本，因为 3.0 的 nn.Hardswish 还没有被支持。
# 修改如下代码以至于可以正确导出openvino模型
## 修改激活函数
由于onnx和openvino 还不支持 Hardswitch，要将 Hardswish 激活函数改成 Relu 或者 Leaky Relu。

```
# yolov5/models/common.py
# Line 26 in 5e0b90d
# self.act = nn.Hardswish() if act else nn.Identity()
self.act = nn.Relu() if act else nn.Identity()
```
## 修改yolo.py

```
# yolov5/models/yolo.py
# Lines 49 to 53 in 5e0b90d
#    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy 
#    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh 
#    z.append(y.view(bs, -1, self.no)) 
#  
# return x if self.training else (torch.cat(z, 1), x) 
```
#### 修改输出层堆叠，不包含输入层

```
    c=(y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
    d=(y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    e=y[..., 4:]
    f=torch.cat((c,d,e),4)
    z.append(f.view(bs, -1, self.no))

  return x if self.training else torch.cat(z, 1)
```
## 修改export.py

```
# yolov5/models/export.py
# Line 31 in 5e0b90d
# model.model[-1].export = True  # set Detect() layer export=True 
model.model[-1].export = False
```
#### 因为版本为10的 opset 能支持 resize 算子，要修改 opset 版本号。

```
# yolov5/models/export.py
# Lines 51 to 52 in 5e0b90d
# torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'], 
torch.onnx.export(model, img, f, verbose=False, opset_version=10, input_names=['images'], 
                   output_names=['classes', 'boxes'] if y is None else ['output'])
```

#### 务必确保 torch=1.15.1，torchvision=0.6.1，onnx==1.7，opset=10。激活函数为 Relu，并修改了网络推理层。
# pth转onnx

```

export PYTHONPATH="$PWD"  
python models/export.py --weights yolov5s.pt --img 640 --batch 1 
```
#### 显示导出为 onnx 和 torchscript 文件即可。

```

ONNX export success, saved as ./yolov5s.onnx
Export complete. Visualize with https://github.com/lutzroeder/netron.
```
# onnx转化为IR

```

python3 /opt/intel/openvino_2020.4.287/deployment_tools/model_optimizer/mo.py 
    --input_model yolov5s_2.0.onnx 
    --output_dir ./out 
    --input_shape [1,3,640,640]
```

顺利的话，就能在 out 目录下生成 yolov5s 的 IR 模型了。

# 基于Python调用IR模型

```
#修改参数匹配训练模型
git clone https://github.com/linhaoqi027/yolov5_openvino_sdk.git
```
修改推理设备和输入 shape

```
# device = 'CPU'
# input_h, input_w, input_c, input_n = (480, 480, 3, 1)
device = 'MYRIAD'
input_h, input_w, input_c, input_n = (640, 640, 3, 1)
```
#### 修改类别信息

```

# label_id_map = {
#     0: "fire",
# }
names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
       'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
       'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
       'hair drier', 'toothbrush']

label_id_map = {index: item for index, item in enumerate(names)}
```
修改多类别输出

```
        for idx, proposal in enumerate(data):
            if proposal[4] > 0:
                print(proposal)
                confidence = proposal[4]
                xmin = np.int(iw * (proposal[0] / 640))
                ymin = np.int(ih * (proposal[1] / 640))
                xmax = np.int(iw * (proposal[2] / 640))
                ymax = np.int(ih * (proposal[3] / 640))
                idx = int(proposal[5])
                #             if label not in label_id_map:
                #                 log.warning(f'{label} does not in {label_id_map}')
                #                 continue
                detect_objs.append({
                    'name': label_id_map[idx],
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'confidence': float(confidence)
                })
```
## 推理输出

```
if __name__ == '__main__':
    # Test API
    img = cv2.imread('../inference/images/bus.jpg')
    predictor = init()
    import time
    t = time.time()
    n = 10
    for i in range(n):
        result = process_image(predictor, img)

    print("平均推理时间",(time.time()-t)/n)
    print("FPS", 1/((time.time()-t)/n))
    # log.info(result)
    for obj in json.loads(result)['objects']:
        print(obj)
```

# 完成代码

```
from __future__ import print_function

import logging as log
import os
import pathlib
import json
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import torch
import torchvision
import time


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    prediction = torch.from_numpy(prediction)
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


device = 'CPU'
# input_h, input_w, input_c, input_n = (480, 480, 3, 1)
input_h, input_w, input_c, input_n = (640, 640, 3, 1)
log.basicConfig(level=log.DEBUG)

# For objection detection task, replace your target labels here.
# label_id_map = {
#     0: "fire",
# }

names=['smoke']

label_id_map = {index: item for index, item in enumerate(names)}

exec_net = None


def init():
    """Initialize model
    Returns: model
    """
    # model_xml = "/project/train/src_repo/yolov5/runs/exp0/weights/best.xml"
    model_xml = "best.xml"
    if not os.path.isfile(model_xml):
        log.error(f'{model_xml} does not exist')
        return None
    model_bin = pathlib.Path(model_xml).with_suffix('.bin').as_posix()
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    # Load Inference Engine
    #     log.info('Loading Inference Engine')
    ie = IECore()
    global exec_net
    exec_net = ie.load_network(network=net, device_name=device)
    #     log.info('Device info:')
    #     versions = ie.get_versions(device)
    #     print("{}".format(device))
    #     print("MKLDNNPlugin version ......... {}.{}".format(versions[device].major, versions[device].minor))
    #     print("Build ........... {}".format(versions[device].build_number))

    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape
    global input_h, input_w, input_c, input_n
    input_h, input_w, input_c, input_n = h, w, c, n

    return net


def process_image(net, input_image):
    """Do inference to analysis input_image and get output
    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        thresh: thresh value
    Returns: process result
    """
    if not net or input_image is None:
        log.error('Invalid input args')
        return None
    #     log.info(f'process_image, ({input_image.shape}')
    ih, iw, _ = input_image.shape

    # --------------------------- Prepare input blobs -----------------------------------------------------
    if ih != input_h or iw != input_w:
        input_image = cv2.resize(input_image, (input_w, input_h))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image / 255
    input_image = input_image.transpose((2, 0, 1))
    images = np.ndarray(shape=(input_n, input_c, input_h, input_w))
    images[0] = input_image

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # --------------------------- Prepare output blobs ----------------------------------------------------
    #     log.info('Preparing output blobs')
    #     log.info(f"The output_name{net.outputs}")
    # print(net.outputs)
    #     output_name = "Transpose_305"
    #     try:
    #         output_info = net.outputs[output_name]
    #     except KeyError:
    #         log.error(f"Can't find a {output_name} layer in the topology")
    #         return None

    #     output_dims = output_info.shape
    #     log.info(f"The output_dims{output_dims}")
    #     if len(output_dims) != 4:
    #         log.error("Incorrect output dimensions for yolo model")
    #     max_proposal_count, object_size = output_dims[2], output_dims[3]

    #     if object_size != 7:
    #         log.error("Output item should have 7 as a last dimension")

    # output_info.precision = "FP32"

    # --------------------------- Performing inference ----------------------------------------------------
    #     log.info("Creating infer request and starting inference")
    res = exec_net.infer(inputs={input_blob: images})

    # --------------------------- Read and postprocess output ---------------------------------------------
    #     log.info("Processing output blobs")

    #     res = res[out_blob]
    data = res[out_blob]

    data = non_max_suppression(data, 0.4, 0.5)
    detect_objs = []

    data = data[0].numpy()
    for idx, proposal in enumerate(data):
        if proposal[4] > 0:
            print(proposal)
            confidence = proposal[4]
            xmin = np.int(iw * (proposal[0] / 640))
            ymin = np.int(ih * (proposal[1] / 640))
            xmax = np.int(iw * (proposal[2] / 640))
            ymax = np.int(ih * (proposal[3] / 640))
            idx = int(proposal[5])
            #             if label not in label_id_map:
            #                 log.warning(f'{label} does not in {label_id_map}')
            #                 continue
            detect_objs.append({
                'name': label_id_map[idx],
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax),
                'confidence': float(confidence)
            })
    return json.dumps({"objects": detect_objs})


if __name__ == '__main__':
    # Test API
    img = cv2.imread('000002.jpg')
    predictor = init()
    result = process_image(predictor, img)
    log.info(result)
```
## 错误解决
1.在Python中导入openvino时报错:from .ie_api import * ImportError: DLL load failed: 找不到指定的模块
https://blog.csdn.net/Thomson617/article/details/101446356
2.RuntimeError: No such operator torchvision::nms问题解决
https://blog.csdn.net/yrwang_xd/article/details/105936538

# 参考链接
*  [【深入YoloV5（开源）】基于YoloV5的模型优化技术与使用OpenVINO推理实现](https://mp.weixin.qq.com/s/m-bn-Q0dhfav-YsI5b-oLg)
* [用树莓派4b构建深度学习应用（九）Yolo篇](https://mp.weixin.qq.com/s/DdsxyGAatsOXGXVMgE9DfQ)