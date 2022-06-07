# 环境安装
* [Jetson nano配置pytorch环境（tensorflow也可）部署yolov5](https://blog.csdn.net/qq_43263543/article/details/115128258?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_v2~rank_aggregation-14-115128258.pc_agg_rank_aggregation&utm_term=jetsonnano%E9%83%A8%E7%BD%B2yolov5&spm=1000.2123.3001.4430)
* [Jetson Nano使用TensorRT加速yolov3-tiny目标识别](https://zhuanlan.zhihu.com/p/344028017)
* [jetson专栏](https://blog.csdn.net/beckhans/category_8839715.html)
* [jetson nano专栏](https://blog.csdn.net/walletiger/category_10584720.html)
* [nano上部署自己训练的yolov5模型（cuda配置,pytorch,tensorrt加速一站式解决）](https://blog.csdn.net/weixin_45454706/article/details/110346822?utm_term=jetsonnano%E9%83%A8%E7%BD%B2yolov5&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-3-110346822&spm=3001.4430)
* [Jetson nano编译安装OpenCV4.1.1和OpenCV_contirb-4.1.1](https://blog.csdn.net/weixin_42640549/article/details/104732567?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159568341719725211901367%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=159568341719725211901367&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_v2~rank_blog_v1-4-104732567.pc_v2_rank_blog_v1&utm_term=jetson%20nano%20opencv%204.1.1&spm=1018.2118.3001.4187)
* [Jetson Nano跑通yolov3（二），禁止桌面](https://blog.csdn.net/alphonse2017/article/details/89634767?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_v2~rank_aggregation-8-89634767.pc_agg_rank_aggregation&utm_term=jetsonnano%E9%83%A8%E7%BD%B2yolov5&spm=1000.2123.3001.4430)
* [Jetson Nano （ 四）OpenCV的编译(4.1与4.4)](https://blog.csdn.net/djj199301111/article/details/107774229)
* [Jetson Nano（ 五） TensorRT yolov4 yolov4-tiny yolov5 实测](https://blog.csdn.net/djj199301111/article/details/110173275)
* [目标检测算法实现（六）——Yolov5实战-Jetson Nano部署](https://blog.csdn.net/qq_40305597/article/details/117320573?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_v2~rank_aggregation-13-117320573.pc_agg_rank_aggregation&utm_term=jetsonnano%E9%83%A8%E7%BD%B2yolov5&spm=1000.2123.3001.4430)
* [Jetson Nano （八） PaddlePaddle 环境配置 PaddleHub—OCR测试](https://blog.csdn.net/djj199301111/article/details/114487656)
* 



## 软件配置
* [Jetson Nano 入门教程4 - 必备软件 Jupyterlab 3.0](https://zhuanlan.zhihu.com/p/342803670)
* [opencv源码安装&卸载](https://blog.csdn.net/alphonse2017/article/details/94624181)

## 人脸
* [jetbot05 人脸检测之 实时 yoloface](https://blog.csdn.net/walletiger/article/details/109784683)
* [jetbot 06 之实时人脸表情检测](https://blog.csdn.net/walletiger/article/details/109837667)
* [jetbot11 之 人手目标检测(hand detect)](https://blog.csdn.net/walletiger/article/details/111027113?spm=1001.2014.3001.5501)



## jetson ncnn
### 环境安装
* [在jetson nano上配置ncnn](https://zhuanlan.zhihu.com/p/285594861)


## 

# torch=1.6
git clone --branch v0.7.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.7.0
sudo python3 setup.py install 
cd ../

# torch=1.7
git clone --branch v0.8.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.8.0
sudo python3 setup.py install 
cd ../

```
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))
c = a + b
print('Tensor c = ' + str(c))
```



# jetson tensorRT

原文链接：https://blog.csdn.net/djj199301111/article/details/110173275

```
系统	模型	框架	输入	精度	时间(ms)
JP4.3	yolov5s v1.0	PyTorch 1.4	416*416	—	60~80
JP4.4.1	yolov4	Tensorrt 7.1.3.0	416*416	FP16	280
JP4.4.1	yolov4	Tensorrt 7.1.3.0	416*416	FP32	380
JP4.4.1	yolov4-tiny(batch 1)	Tensorrt 7.1.3.0	416*416	FP16	45
JP4.4.1	yolov4-tiny(batch 4)	Tensorrt 7.1.3.0	416*416	FP16	38
JP4.4.1	yolov5s v3.1	Tensorrt 7.1.3.0	608*608	FP16	110~130
JP4.4.1	yolov5s v6.0	Tensorrt 7.1.3.0	608*608	FP16	70-78
JP4.4.1	yolov5n v6.0	Tensorrt 7.1.3.0	608*608	FP16	35-47
JP4.4.1	yolov5s v3.1	PyTorch 1.6	608*608	—	170~210
JP4.4.1	yolov5s v6.0	PyTorch 1.8	608*608	—	140~175
JP4.4.1	yolov5s v3.1	PyTorch 1.6	416*416	—	120~140
```


### jetson使用教程
```python
sudo pip3 install -U jetson-stats

sudo jtop
```




