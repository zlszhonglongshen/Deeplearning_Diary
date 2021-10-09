### docker pull命令
docker pull tensorflow/serving:latest-devel

### docker 命令行
docker run -p 8501:8501 --mount type=bind,\
source=/root/zhongls/keras-and-tensorflow-serving-master/my_image_classifier,\
target=/models/ImageClassifier \
-e MODEL_NAME=ImageClassifier \
-t tensorflow/serving &

### 使用上面的docker命令启动TF Server ：
(1)-p 8501:8501是端口映射，是将容器的8501端口映射到宿主机的8501端口，后面预测的时候使用该端口；
(2)-e MODEL_NAME=bert 设置模型名称；
(3)--mount type=bind,source=/opt/developer/wp/learn/bert, target=/models/bert 是将宿主机的路径/opt/developer/wp/learn/bert 挂载到容器的/models/bert 下。
/opt/developer/wp/learn/bert是通过上述py脚本生成的Saved_model的路径。容器内部会根据绑定的路径读取模型文件；


### docker测试脚本
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
  -X POST http://localhost:8501/v1/models/half_plus_two:predict
  
 

##################tensorflow-serving部署过程##################
1：将模型转换成tensorflow-serving需要的格式

2：启动tensorflow-serving
docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=/opt/zhongls/pb/cigar/,target=/models/cigar --mount type=bind,source=/opt/zhongls/model_zoo-master/TensorFlow/mtcnn/export_model_for_serving,target=/models/mtcnn --mount type=bind,source=/opt/zhongls/pb/cigardetect,target=/models/cigardetect --mount type=bind,source=/opt/zhongls/pb/handdetect,target=/models/handdetect --mount type=bind,source=/opt/zhongls/keras-and-tensorflow-serving-master/models,target=/models/facesNum --mount type=bind,source=/opt/zhongls/keras-and-tensorflow-serving-master/selfie,target=/models/selfie --mount type=bind,source=/opt/zhongls/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,target=/models/half_plus_two --mount type=bind,source=/opt/zhongls/pb/ocrlstm,target=/models/ocrlstm --mount type=bind,source=/opt/zhongls/pb/ocrdetect,target=/models/ocrdetect --mount type=bind,source=/opt/zhongls/pb/angledetect,target=/models/angledetect --mount type=bind,source=/opt/zhongls/pb/model.config,target=/models/model.config -t tensorflow/serving:latest --model_config_file=/models/model.config &

3.配置proto&server.py文件
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. calculate.proto

4.配置calculate.py文件

5.启动文件
docker run --expose 5011 -d -p 5564:5011  -v /opt/zhongls/hand_proto/:/root/app/ zls/anacondaocr:v2  python  /root/app/server.py



##################参考文章链接##################
* [TensorFlow-Serving的使用实战案例笔记（tf=1.4）](https://blog.csdn.net/sinat_26917383/article/details/104902909)
* [TensorFlow Serving + Docker + Tornado机器学习模型生产级快速部署](https://zhuanlan.zhihu.com/p/52096200)
* [TensorFlow-Serving入门,restful教程](https://springzzj.github.io/2020/01/17/TensorFlow-Serving%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/)
* [使用Grpc调用TensorFlow-Serving服务](https://zhuanlan.zhihu.com/p/110727286)
* [使用 TensorFlow Serving 和 Docker 快速部署机器学习服务,比较完整](https://blog.csdn.net/weixin_34343000/article/details/88118667)
* [tensorflow-serving的docker方式同时加载多个模型](https://www.jianshu.com/p/b1438bdcb121)
* [tensorflow tfserving 部署多个模型、使用不同版本的模型](https://blog.csdn.net/JerryZhang__/article/details/86516428)
* [使用docker和tf serving搭建模型预测服务,grpc](https://blog.csdn.net/JerryZhang__/article/details/85107506)
* [tensorflow模型部署，并针对多输入的情况进行讨论](https://blog.csdn.net/u011734144/article/details/82107610)
* [tensorflow模型部署](https://www.jianshu.com/p/d11a5c3dc757)
* [MTCNN 的 TensorFlow Serving](https://blog.csdn.net/FortiLZ/article/details/81396683?utm_source=blogxgwz6&tdsourcetag=s_pctim_aiomsg)
* [Tensorflow Serving部署tensorflow、keras模型详解](https://blog.csdn.net/u010159842/article/details/88529281)
* [tf_serving-模型训练、导出、部署（解析）](https://blog.csdn.net/wc781708249/article/details/78606514)
* [分类模型，tensorflow+tensorflow-serving+docker+grpc](https://blog.csdn.net/u013714645/article/details/81449487?utm_source=blogxgwz1)
* [如何将自己的模型转换为Tensorflow-list可用模型](https://mp.weixin.qq.com/s/AIQtlNNEb0lKyshGZcpZAw)




##################mlflow参考文章链接##################
* [mlflow model](https://blog.csdn.net/chenghouxian7338/article/details/101034156)





