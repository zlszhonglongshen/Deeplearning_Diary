# coding: utf-8
# author: hxy
# 2019-12-10
"""
视频流的推理过程；
默认推理过程在CPU上；
"""

import cv2
import time
import logging
import numpy as np
import onnxruntime
from lib.darknet_api import get_boxes


# 定义日志格式
def log_set():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(filename)s：%(lineno)d - %(levelname)s - %(message)s')


# 加载onnx模型pip
def load_model(onnx_model):
    sess = onnxruntime.InferenceSession(onnx_model)
    in_name = [input.name for input in sess.get_inputs()][0]
    out_name = [output.name for output in sess.get_outputs()]
    logging.info("输入的name:{}, 输出的name:{}".format(in_name, out_name))

    return sess, in_name, out_name


def frame_process(frame, input_shape=(608, 608)):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


# 视屏预处理
def stream_inference():
    # 基本的参数设定
    label = ["background", "person",
	        "bicycle", "car", "motorbike", "aeroplane",
	        "bus", "train", "truck", "boat", "traffic light",
	        "fire hydrant", "stop sign", "parking meter", "bench",
	        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
	        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
	        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
	        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
	        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
	        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
	        "scissors", "teddy bear", "hair drier", "toothbrush"]
    anchors_yolo_tiny = [[(81, 82), (135, 169), (344, 319)], [(10, 14), (23, 27), (37, 58)]]
    anchors_yolo = [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)]]
    session, in_name, out_name = load_model(onnx_model='yolov3_608.onnx')

    # rtsp = ''
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        input_shape = frame.shape
        s = time.time()
        test_data = frame_process(frame, input_shape=(608, 608))
        logging.info("process per pic spend time is:{}ms".format((time.time() - s)*1000))
        s1 = time.time()
        prediction = session.run(out_name, {in_name: test_data})
        s2 = time.time()
        print("prediction cost time: %.3fms" % (s2 - s1))
        boxes = get_boxes(prediction=prediction,
                          anchors=anchors_yolo,
                          img_shape=(608, 608))
        print("get box cost time:{}ms".format((time.time()-s2)*1000))
        for box in boxes:
            x1 = int((box[0] - box[2] / 2) * input_shape[1])
            y1 = int((box[1] - box[3] / 2) * input_shape[0])
            x2 = int((box[0] + box[2] / 2) * input_shape[1])
            y2 = int((box[1] + box[3] / 2) * input_shape[0])
            logging.info(label[int(box[5])] + ":" + str(round(box[4], 3)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, label[int(box[5])] + ":" + str(round(box[4], 3)),
                        (x1 + 5, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1)

        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow("Results", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    log_set()
    stream_inference()
