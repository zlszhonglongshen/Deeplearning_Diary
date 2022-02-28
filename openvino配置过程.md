* 添加环境变量PATH
  1：C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\inference_engine\bin\intel64\Release
  2：C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\python\python3.6\openvino\inference_engine

* 复制C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\python\python3.6\openvino 到python36的lib-sitebages

* 进入到工程下
  python convert_weights_pb.py --class_names coco.names --weights_file yolov3-tiny.weights --data_format NHWC --tiny --output_graph frozen_tiny_yolo_v3.pb

* 进入到C:\Windows\System32\cmd.exe
  python mo_tf.py --input_model frozen_tiny_yolo_v3.pb  --tensorflow_use_custom_operations_config yolo_v3_tiny_changed.json  --input_shape=[1,416,416,3]  --data_type=FP32


# model server案例
docker run -v /root/zls/model/models:/models:ro -p 9000:9000 openvino/model_server:latest --model_path /models/model1 --model_name face-detection --port 9000 --log_level DEBUG --shape auto

docker run -v /models:/models:ro -p 9000:9000 openvino/model_server:latest --model_path /models/model1 --model_name face-detection --port 9000 --log_level DEBUG --shape auto


# 参考链接：
* [OpenVINO Model Server,教程](https://zhuanlan.zhihu.com/p/102107664)

* [OpenVINO Model Server的服务化部署——step1（OpenVINO™ Model Server Quickstart）](https://www.cnblogs.com/jsxyhelu/p/13796161.html)

* [【玩转YOLOv5】YOLOv5转openvino并进行部署](https://blog.csdn.net/weixin_44936889/article/details/110940322?spm=1001.2014.3001.5501)
* [基于openvino 2019R3的INT8推理(inference)性能的深入研究 (一) MobilenetV2](https://blog.csdn.net/sandmangu/article/details/105555046?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-11-105555046.nonecase&utm_term=openvino%20%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96&spm=1000.2123.3001.4430)
* [【深入YoloV5（开源）】基于YoloV5的模型优化技术与使用OpenVINO推理实现](https://blog.csdn.net/qq_46098574/article/details/109702222?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-13-109702222.nonecase&utm_term=openvino%20%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96&spm=1000.2123.3001.4430)
* [用树莓派4b构建深度学习应用（九）Yolo篇](https://mp.weixin.qq.com/s/DdsxyGAatsOXGXVMgE9DfQ)
* [自建网络训练后，openVINO部署记录（win10）-部署DBFACE](https://blog.csdn.net/Ai_Smith/article/details/109063005?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-12-109063005.nonecase&utm_term=openvino%20%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96&spm=1000.2123.3001.4430)
* [YOLOv3-tiny在VS2015上使用Openvino部署](https://blog.csdn.net/just_sort/article/details/103454047?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-19-103454047.nonecase&utm_term=openvino%20%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96&spm=1000.2123.3001.4430)
* [openvino使用（一）转换并量化（INT8）分类网络模型](https://blog.csdn.net/qq_37541097/article/details/108382807)
* [openvino 加速yolo-tiny目标检测(windows)](https://blog.csdn.net/weixin_44683176/article/details/104967807?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-14-104967807.nonecase&utm_term=openvino%20%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96&spm=1000.2123.3001.4430)
* https://blog.csdn.net/sinat_35907936/article/details/88760618
* https://cloud.tencent.com/developer/article/1424650
* https://blog.csdn.net/yuanlulu/article/details/86619099
* [yolov5-openvino](https://mp.weixin.qq.com/s/DdsxyGAatsOXGXVMgE9DfQ)
* [安装 OpenVINO](https://blog.csdn.net/lemon4869/article/details/107145684)
* [编写 OpenVINO 应用程序（C++ 版）](https://blog.csdn.net/lemon4869/article/details/107145841)
* [openvino系列推理教程](https://blog.csdn.net/sandmangu/category_9585303.html)
* [OpenVINO安装记录（Ubuntu18.04）](https://blog.csdn.net/qq_37541097/article/details/108374828)
* [OpenVINO于linux无UI无sudo权限、全命令行安装记录](https://blog.csdn.net/github_28260175/article/details/106277248?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-17-106277248.nonecase&utm_term=openvino%20%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96&spm=1000.2123.3001.4430)





# v5转openvino

* [v5转ncnn](https://github.com/sunnyden/YOLOV5_NCNN_Android/issues/8)