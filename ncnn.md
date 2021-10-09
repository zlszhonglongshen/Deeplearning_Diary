# NCNN

### 配置相关

[Windows下ncnn环境配置（VS2019）](https://blog.csdn.net/qq_36890370/article/details/104966786)



### 相关案例

[【手把手AI项目】七、MobileNetSSD通过Ncnn前向推理框架在PC端的使用(目标检测 objection detection)](https://blog.csdn.net/qq_33431368/article/details/84990390)

[onnx2ncnn并在pc端调用ncnn模型](https://blog.csdn.net/qq_36113487/article/details/100676205)

[记mobilenet_v2的pytorch模型转onnx模型再转ncnn模型一段不堪回首的历程](https://blog.csdn.net/weixin_42184622/article/details/102593448)

[ncnn android算法移植（五）——DBFace移植](https://blog.csdn.net/u011622208/article/details/106275495)

[记mobilenet_v2的pytorch模型转onnx模型再转ncnn模型一段不堪回首的历程](https://blog.csdn.net/weixin_42184622/article/details/102593448)

[深度学习模型移植pytorch->onnx->ncnn->android](https://blog.csdn.net/hmzjwhmzjw/article/details/94027816)

[Pytorch转NCNN的流程记录-chineseocr-lite](https://zhuanlan.zhihu.com/p/124294444)

[RetinaFace Pytorch实现训练、测试，pytorch模型转onnx转ncnn C++推理](https://blog.csdn.net/qq_38109843/article/details/104433933)

* [[yolov5＞onnx＞ncnn＞apk](https://my.oschina.net/u/3337401/blog/4708237)](https://my.oschina.net/u/3337401/blog/4708237)



# 命令行

onnx2ncnn
python -m onnxsim  last.onnx yolov5s-sim.onnx

onnx2ncnn  yolov5s-sim.onnx yolov5.param yolov5.bin

ncnnoptimize yolact.param yolact.bin yolact-opt.param yolact-opt.bin 0

