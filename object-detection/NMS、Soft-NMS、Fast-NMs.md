## NMS

![浅谈NMS的多种实现](https://pic4.zhimg.com/v2-b24ea7df07b91341dc3e5af6c555f83b_1440w.jpg?source=172ae18b)

NMS：先对每个框的score进行排序，首先选择第一个，也就是score最高的框，它一定是我们要保留的框，然后拿他和剩下的框进行比较，如果IOU大于一定的阈值，说明两者重合度高，应该去掉，这样筛选出的框就是和第一个框重合度低的框，第一次迭代结束。第二次从保留的框中选出score第一的框，重复上述过程指导没有框保留了。

```
def nms(self, bboxes, scores, threshold=0.5):
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]
        areas = (x2-x1)*(y2-y1)   # [N,]
        _, order = scores.sort(0, descending=True)    # 降序

        keep = []
        while order.numel() > 0:       # torch.numel()返回张量元素个数
            if order.numel() == 1:     # 保留框只剩一个
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()    # 保留scores最大的那个框box[i]
                keep.append(i)

            # 计算box[i]与其余各框的IOU
            xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

            iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
            idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
            if idx.numel() == 0:
                break
            order = order[idx+1]  # 修补索引之间的差值
        return torch.LongTensor(keep) 

```

## Soft-NMS

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205153327863.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2NTMwOTky,size_16,color_FFFFFF,t_70#pic_center)

如果按照上述传统的NMS算法，红色框比绿色框置信度高，所以排序后会先处理红色框，而此时在计算与其他框的IOU时，绿色框和红色框的IOU大于阈值，从而会排除掉绿色框，同时NMS的阈值也难以确定，设置搞了会误检，设置低了会漏检，如上图所示的情况。而Soft-NMS的思路就是将所有IOU大于阈值的框降低其置信度，而不是删除。

算法如下图所示，

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205155653770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2NTMwOTky,size_16,color_FFFFFF,t_70#pic_center)

传统的NMS可以描述为将IOU大于阈值的框的得分置为0：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205155918532.png#pic_center)

而Soft-NMS提出了两种方式，一种是线性加权，一种是高斯加权：

线性加权：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205160153160.png#pic_center)

高斯加权

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120516021780.png#pic_center)

实验结果如下图所示，可以看出来在不增加额外的计算量下可以平均提升1%的精度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205160700268.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2NTMwOTky,size_16,color_FFFFFF,t_70#pic_center)

## Fast-NMS

该算法是在实例分割论文[YOLACT](https://arxiv.org/pdf/1904.02689.pdf)中所提出，此处为[github](https://github.com/dbolya/yolact)链接。该论文主打实时(`fps`>30)，说传统的`NMS`可以利用矩阵简化从而降低时间，但不得不牺牲一些精度，实验结果显示虽然降低了0.1mAP，但时间比基于`Cython`实现的`NMS`快11.8ms。

算法代码如下

```
def fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        '''
        boxes:  torch.Size([num_dets, 4])
        masks:  torch.Size([num_dets, 32])
        scores: torch.Size([num_classes, num_dets])
        '''
        # step1: 每一类的框按照scores降序排序后取前top_k个
        scores, idx = scores.sort(1, descending=True) 
        # scores为降序排列 
        # idx为原顺序的索引 
        idx = idx[:, :top_k].contiguous() # 取前top_k个框 
        scores = scores[:, :top_k] 
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4) # torch.Size([num_classes, num_dets, 4])
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1) # torch.Size([num_classes, num_dets, 32]) 其中32为生成的系数个数
        # step2: 计算每一类中，box与box之间的IoU
        iou = jaccard(boxes, boxes) # torch.Size([num_classes, num_dets, num_dets])
        iou.triu_(diagonal=1) # triu_()取上三角 tril_()取下三角 此处将矩阵的下三角和对角线元素删去
        iou_max, _ = iou.max(dim=1) # 按列取大值 torch.Size([num_classes, num_dets])

        # 过滤掉iou大于阈值的框 
        keep = (iou_max <= iou_threshold) # torch.Size([num_classes, num_dets])

        if second_threshold: # 保证保留的框满足一定的置信度
            keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        '''
        tensor([[ 0,  0,  0,  ...,  0,  0,  0],
        [ 1,  1,  1,  ...,  1,  1,  1],
        [ 2,  2,  2,  ...,  2,  2,  2],
        ...,
        [77, 77, 77,  ..., 77, 77, 77],
        [78, 78, 78,  ..., 78, 78, 78],
        [79, 79, 79,  ..., 79, 79, 79]])
        '''
        classes = classes[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx]e
        boxes = boxes[idx]
        masks = masks[idx]
        return boxes, masks, classes, scores # torch.Size([max_num_detections])

```

![image-20211103104513075](/Users/zhongls/Library/Application Support/typora-user-images/image-20211103104513075.png)

舍弃超过阈值的框，假设阈值为0.5，那么该类中bbox2和bbox3都要被舍弃，只留下bbox1。因为首先预测框已经按得分降序排列好了，并且每一个元素都是行号小于列号，元素大于阈值就代表着这一列对应的框与比它置信度高的框过于重叠，比如0.7，就代表着bbox2和bbox1过于重叠，并且bbox1的置信度较高，应该排除bbox2。但是如果按照传统的`NMS`方法，bbox2会被排除，bbox3会被保留。所以`fast-NMS`会去掉更多的框，但是作者认为速度至上，并且实验也证明精度下降一点点速度却大大提高，如下图所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205202438232.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2NTMwOTky,size_16,color_FFFFFF,t_70#pic_center)

参考博客：
[1] [NMS](https://zhuanlan.zhihu.com/p/54709759)
[2] [Soft-NMS](https://blog.csdn.net/app_12062011/article/details/77963494)
[3] [fast-NMS](https://zhuanlan.zhihu.com/p/76470432)

https://zhuanlan.zhihu.com/p/157900024
