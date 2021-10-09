[net]
# Testing #测试模式
batch=1
subdivisions=1
# Training #训练模式 每次前向图片的数目=batch/subdivisions
# batch=64
# subdivisions=16
#关于batch与subdivision：在训练输出中，训练迭代包括8组，这些batch样本又被平均分成subdivision=8次送入网络参与训练，以减轻内存占用的压力；batch越大，训练效果越好，subdivision越大，占用内存压力越小

width=416
height=416
channels=3
#网络输入的宽、高、通道数这三个参数中，要求width==height, 并且为32的倍数，大分辨率可以检测到更加细小的物体，从而影响precision

momentum=0.9 #动量，影响梯度下降到最优的速度，一般默认0.9
decay=0.0005 #权重衰减正则系数，防止过拟合
angle=0 #旋转角度，从而生成更多训练样本
saturation = 1.5 #调整饱和度，从而生成更多训练样本
exposure = 1.5 #调整曝光度，从而生成更多训练样本
hue=.1 #调整色调，从而生成更多训练样本

learning_rate=0.001
#学习率决定了权值更新的速度，学习率大，更新的就快，但太快容易越过最优值，而学习率太小又更新的慢，效率低，一般学习率随着训练的进行不断更改，先高一点，然后慢慢降低，一般在0.01--0.001

burn_in=1000
#学习率控制的参数，在迭代次数小于burn_in时，其学习率的更新有一种方式，大于burn_in时，才采用policy的更新方式
max_batches = 50200
#迭代次数，1000次以内，每训练100次保存一次权重，1000次以上，每训练10000次保存一次权重
policy=steps # 学习率策略，学习率下降的方式
steps=40000,45000 #学习率变动步长
scales=.1,.1
#学习率变动因子：如迭代到40000次时，学习率衰减十倍，45000次迭代时，学习率又会在前一个学习率的基础上衰减十倍

[convolutional]
batch_normalize=1 #BN
filters=32 #卷积核数目
size=3 #卷积核尺寸
stride=1 #做卷积运算的步长
pad=1
#如果pad为0,padding由 padding参数指定；如果pad为1，padding大小为size/2，padding应该是对输入图像左边缘拓展的像素数量

activation=leaky #激活函数类型


[yolo]
mask = 6,7,8 #使用anchor时使用前三个尺寸
anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
#anchors是可以事先通过cmd指令计算出来的，是和图片数量，width,height以及cluster(就是下面的num的值，即想要使用的anchors的数量)相关的预选框，可以手工挑选，也可以通过k-means算法从训练样本中学出

classes=20
num=9
#每个grid cell预测几个box,和anchors的数量一致。当想要使用更多anchors时需要调大num，且如果调大num后训练时Obj趋近0的话可以尝试调大object_scale
jitter=.3 #通过抖动来防止过拟合,jitter就是crop的参数
ignore_thresh = .5
#ignore_thresh 指得是参与计算的IOU阈值大小。当预测的检测框与ground true的IOU大于ignore_thresh的时候，参与loss的计算，否则，检测框的不参与损失计算，目的是控制参与loss计算的检测框的规模，当ignore_thresh过于大，接近于1的时候，那么参与检测框回归loss的个数就会比较少，同时也容易造成过拟合；而如果ignore_thresh设置的过于小，那么参与计算的会数量规模就会很大。同时也容易在进行检测框回归的时候造成欠拟合。
#参数设置：一般选取0.5-0.7之间的一个值，之前的计算基础都是小尺度（13*13）用的是0.7，（26*26）用的是0.5。这次先将0.5更改为0.7。
truth_thresh = 1
random=1 #如果显存小，设置为0，关闭多尺度训练，random设置成1，可以增加检测精度precision，每次迭代图片大小随机从320到608，步长为32，如果为0，每次训练大小与输入大小一致





batch：每次迭代要进行训练的图片数量

subdivision：batch中的图片再产生子集，源码中的图片数量int imgs = net.batch * net.subdivisions * ngpus

也就是说每轮迭代会从所有训练集里随机抽取 batch = 64 个样本参与训练，每个batch又会被分成 64/16 = 4 次送入网络参与训练，以减轻内存占用的压力，也就是每次 subdivision = 16 送入网络

width：输入图片宽度， height：输入图片高度，channels ：输入图片通道数

对于每次迭代训练，YOLO会基于角度(angle)，饱和度(saturation)，曝光(exposure)，色调(hue)产生新的训练图片

angle：图片角度变化，单位为度，假如 angle=5，就是生成新图片的时候随机旋转-5~5度

weight decay：权值衰减

防止过拟合，当网络逐渐过拟合时网络权值往往会变大，因此，为了避免过拟合，在每次迭代过程中以某个小因子降低每个权值，也等效于给误差函数添加一个惩罚项，常用的惩罚项是所有权重的平方乘以一个衰减常量之和。权值衰减惩罚项使得权值收敛到较小的绝对值。

angle：图片角度变化，单位为度，假如 angle=5，就是生成新图片的时候随机旋转-5~5度

saturation & exposure: 饱和度与曝光变化大小，tiny-yolo-voc.cfg中1到1.5倍，以及1/1.5~1倍

hue：色调变化范围，tiny-yolo-voc.cfg中-0.1~0.1

max_batches：最大迭代次数

learning rate：学习率

学习率决定了参数移动到最优值的速度快慢，如果学习率过大，很可能会越过最优值导致函数无法收敛，甚至发散；反之，如果学习率过小，优化的效率可能过低，算法长时间无法收敛，也易使算法陷入局部最优（非凸函数不能保证达到全局最优）。合适的学习率应该是在保证收敛的前提下，能尽快收敛。
设置较好的learning rate，需要不断尝试。在一开始的时候，可以将其设大一点，这样可以使weights快一点发生改变，在迭代一定的epochs之后人工减小学习率。

policy：调整学习率的策略

调整学习率的policy，有如下policy：CONSTANT, STEP, EXP, POLY，STEPS, SIG, RANDOM

steps：学习率变化时的迭代次数

根据batch_num调整学习率，若steps=100,20000,30000，则在迭代100次，20000次，30000次时学习率发生变化，该参数与policy中的steps对应

scales：学习率变化的比率

相对于当前学习率的变化比率，累计相乘，与steps中的参数个数保持一致



# 二、训练时的参数含义
397： 指示当前训练的迭代次数
7.748884： 是总体的Loss(损失）
7.593599 avg： 是平均Loss，这个数值应该越低越好，一般来说，一旦这个数值低于0.060730 avg就可以终止训练了。
0.000025 rate： 代表当前的学习率，是在.cfg文件中定义的。
5.510836 seconds： 表示当前批次训练花费的总时间。
25408 images： 这一行最后的这个数值是397*64的大小，表示到目前为止，参与训练的图片的总量。
Avg IOU: 0.574694： 表示在当前subdivision内的图片的平均IOU，代表预测的矩形框和真实目标的交集与并集之比，这里是57.46%，这个模型需要进一步的训练。
Class: 0.962937： 标注物体分类的正确率，期望该值趋近于1。
Obj: 0.022145： 越接近1越好。
No Obj: 0.000459： 期望该值越来越小，但不为零。
count: 2：count后的值是所有的当前subdivision图片（本例中一共8张）中包含正样本的图片的数量。在输出log中的其他行中，可以看到其他subdivision也有的只含有<16个正样本，说明在subdivision中含有不含检测对象的图片。


###参考文档
https://www.cnblogs.com/answerThe/p/11544361.html



