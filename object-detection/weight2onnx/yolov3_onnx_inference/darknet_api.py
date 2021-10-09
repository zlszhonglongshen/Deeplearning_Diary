# coding: utf-8
# 2019-12-10
"""
YOlo相关的预处理api；
"""
import cv2
import time
import numpy as np



# 加载label names；
def get_labels(names_file):
    names = list()
    with open(names_file, 'r') as f:
        lines = f.read()
        for name in lines.splitlines():
            names.append(name)
    f.close()
    return names


# 照片预处理
def process_img(img_path, input_shape):
    ori_img = cv2.imread(img_path)
    img = cv2.resize(ori_img, input_shape)
    image = img[:, :, ::-1].transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :] / 255
    image = np.array(image, dtype=np.float32)
    return ori_img, ori_img.shape, image


# 视频预处理
def frame_process(frame, input_shape):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


# sigmoid函数
def sigmoid(x):
    s = 1 / (1 + np.exp(-1 * x))
    return s


# 获取预测正确的类别，以及概率和索引;
def get_result(class_scores):
    class_score = 0
    class_index = 0
    for i in range(len(class_scores)):
        if class_scores[i] > class_score:
            class_index += 1
            class_score = class_scores[i]
    return class_score, class_index


# 通过置信度筛选得到bboxs
def get_bbox(feat, anchors, image_shape, confidence_threshold=0.25):
    box = list()
    for i in range(len(anchors)):
        for cx in range(feat.shape[0]):
            for cy in range(feat.shape[1]):
                #根据自己的情况来修改
#                 tx = feat[cx][cy][0 + 85 * i]
#                 ty = feat[cx][cy][1 + 85 * i]
#                 tw = feat[cx][cy][2 + 85 * i]
#                 th = feat[cx][cy][3 + 85 * i]
#                 cf = feat[cx][cy][4 + 85 * i]
#                 cp = feat[cx][cy][5 + 85 * i:85 + 85 * i]
                
                tx = feat[cx][cy][0 + 6 * i]
                ty = feat[cx][cy][1 + 6 * i]
                tw = feat[cx][cy][2 + 6 * i]
                th = feat[cx][cy][3 + 6 * i]
                cf = feat[cx][cy][4 + 6 * i]
                cp = feat[cx][cy][5 + 6 * i:6 + 6 * i]

                bx = (sigmoid(tx) + cx) / feat.shape[0]
                by = (sigmoid(ty) + cy) / feat.shape[1]
                bw = anchors[i][0] * np.exp(tw) / image_shape[0]
                bh = anchors[i][1] * np.exp(th) / image_shape[1]
                b_confidence = sigmoid(cf)
                b_class_prob = sigmoid(cp)
                b_scores = b_confidence * b_class_prob
                b_class_score, b_class_index = get_result(b_scores)

                if b_class_score >= confidence_threshold:
                    box.append([bx, by, bw, bh, b_class_score, b_class_index])
    return box


# 采用nms算法筛选获取到的bbox
def nms(boxes, nms_threshold=0.6):
    l = len(boxes)
    if l == 0:
        return []
    else:
        b_x = boxes[:, 0]
        b_y = boxes[:, 1]
        b_w = boxes[:, 2]
        b_h = boxes[:, 3]
        scores = boxes[:, 4]
        areas = (b_w + 1) * (b_h + 1)
        order = scores.argsort()[::-1]
        keep = list()
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(b_x[i], b_x[order[1:]])
            yy1 = np.maximum(b_y[i], b_y[order[1:]])
            xx2 = np.minimum(b_x[i] + b_w[i], b_x[order[1:]] + b_w[order[1:]])
            yy2 = np.minimum(b_y[i] + b_h[i], b_y[order[1:]] + b_h[order[1:]])

            # 相交面积,不重叠时面积为0
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 相并面积,面积1+面积2-相交面积
            union = areas[i] + areas[order[1:]] - inter
            # 计算IoU：交 /（面积1+面积2-交）
            IoU = inter / union
            # 保留IoU小于阈值的box
            inds = np.where(IoU <= nms_threshold)[0]
            order = order[inds + 1]  # 因为IoU数组的长度比order数组少一个,所以这里要将所有下标后移一位

        final_boxes = [boxes[i] for i in keep]
        return final_boxes


# 绘制预测框
def draw_box(boxes, img, img_shape):
    label = ["background", "face"]
    for box in boxes:
        x1 = int((box[0] - box[2] / 2) * img_shape[1])
        y1 = int((box[1] - box[3] / 2) * img_shape[0])
        x2 = int((box[0] + box[2] / 2) * img_shape[1])
        y2 = int((box[1] + box[3] / 2) * img_shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label[int(box[5])] + ":" + str(round(box[4], 3)), (x1 + 5, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
        print(label[int(box[5])] + ":" + "概率值：%.3f" % box[4])
    cv2.imshow('image', img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()


# 获取预测框
def get_boxes(prediction, anchors, img_shape, confidence_threshold=0.25, nms_threshold=0.6):
    boxes = []
    for i in range(len(prediction)):
        feature_map = prediction[i][0].transpose((2, 1, 0))
        box = get_bbox(feature_map, anchors[i], img_shape, confidence_threshold)
        boxes.extend(box)
    Boxes = nms(np.array(boxes), nms_threshold)
    return Boxes
