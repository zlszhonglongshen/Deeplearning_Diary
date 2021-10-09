参考链接:

* [mtcnn从头到尾的详解](https://zhuanlan.zhihu.com/p/58825924)

++++++++++++++++++影响模型速度的想法++++++++++++++++++++

\* 输入推理的图片大小做个限制，图片越大需要做的nms操作量越多，得到的图像金字塔越厚

\* 如果只是想要得到bbox或者key point 二者之一，那么可以考虑在推理的时候把另一半的分支给干掉，这样应该也可以做到加速



![img](https://pic2.zhimg.com/80/v2-8e2d0268a6105c44af8512fbf24a3321_720w.jpg)



流程主要分为以下几个步骤：

1. **首先对test图片不断进行Resize，得到图片金字塔。**按照resize_factor(如0.70，这个具体根据数据集人脸大小分布来确定，基本确定在0.70-0.80之间会比较合适，设的比较大，容易延长推理时间，小了容易漏掉一些中小型人脸）对test图片进行resize，直到大等于Pnet要求的12**12大小。这样子你会得到原图、原图**resize_factor、原图**resize_factor^2...、原图**resize_factor^n（注，最后一个的图片大小会大等于12）这些不同大小的图片，堆叠起来的话像是金字塔，简单称为图片金字塔。注意，这些图像都是要一幅幅输入到Pnet中去得到候选的。

2. **图片金字塔输入Pnet，得到大量的候选（candidate）。**根据上述步骤1得到的图片金字塔，将所有图片输入到Pnet，得到输出map形状是（m, n, 16）。根据分类得分，筛选掉一大部分的候选，再根据得到的4个偏移量对bbox进行校准后得到bbox的左上右下的坐标（根据偏移量矫正先埋个坑，描述训练阶段的时候补），对这些候选根据IOU值再进行非极大值抑制（NMS）筛选掉一大部分候选。详细的说就是根据分类得分从大到小排，得到（num_left, 4）的张量，即num_left个bbox的左上、右下绝对坐标。每次以队列里最大分数值的bbox坐标和剩余坐标求出iou，干掉iou大于0.6（阈值是提前设置的）的框，并把这个最大分数值移到最终结果。重复这个操作，会干掉很多有大量overlap的bbox，最终得到(num_left_after_nms, 16)个候选，这些候选需要根据bbox坐标去原图截出图片后，resize为24*24输入到Rnet。

3. ***经过Pnet筛选出来的候选图片，经过Rnet进行精调。***根据Pnet输出的坐标，去原图上截取出图片（截取图片有个细节是需要截取bbox最大边长的正方形，这是为了保障resize的时候不产生形变和保留更多的人脸框周围细节），resize为24*24，输入到Rnet，进行精调。Rnet仍旧会输出二分类one-hot2个输出、bbox的坐标偏移量4个输出、landmark10个输出，根据二分类得分干掉大部分不是人脸的候选、对截图的bbox进行偏移量调整后（说的简单点就是对左上右下的x、y坐标进行上下左右调整），再次重复Pnet所述的IOU NMS干掉大部分的候选。最终Pnet输出的也是（num_left_after_Rnet, 16），根据bbox的坐标再去原图截出图片输入到Onet，同样也是根据最大边长的正方形截取方法，避免形变和保留更多细节。

4. ***经过Rnet干掉很多候选后的图片输入到Onet，输出准确的bbox坐标和landmark坐标。***大体可以重复Pnet的过程，不过有区别的是这个时候我们除了关注bbox的坐标外，也要输出landmark的坐标。（有小伙伴会问，前面不关注landmark的输出吗？嗯，作者认为关注的很有限，前面之所以也有landmark坐标的输出，主要是希望能够联合landmark坐标使得bbox更精确，换言之，推理阶段的Pnet、Rnet完全可以不用输出landmark，Onet输出即可。当然，训练阶段Pnet、Rnet还是要关注landmark的）经过分类筛选、框调整后的NMS筛选，好的，至此我们就得到准确的人脸bbox坐标和landmark点了，任务完满结束。

OK，将推理阶段的工作流程描述完了，还埋了几个坑：

\- Pnet得到的bbox的偏移量是针对什么的偏移量？我如何得到校准后的bbox坐标？

\- 偏移量矫正是个什么鬼？怎么做？提前回答下：分为bbox的偏移量矫正和landmark的偏移量矫正。

以上的坑，将在后面描述后面阶段的时候填，且听我慢慢道来。



##  训练过程

讲完推理过程，这个时候大家应该会有一个整体的理解了，大体是图片输入到三个网络，每个网络进行精调后输出bbox、landmark，可是你是咋训练的呢，有点匪夷所思。

其实是这样的，训练阶段的图片可不是上面那个老爷爷那么大的图片，而是每个网络都有自己想要的大小的图片即训练集，Pnet是12**12**3， Rnet是24**24**3， Onet是48**48**3的图片。然后每个网络都是单独训练的，即训练好Pnet后训练Rnet，再训练Onet，前者的训练输出是后者的训练输入。

![img](https://pic3.zhimg.com/80/v2-e6c6e11b57eae121c8134408bd0c955a_720w.jpg)



所以我们现在来观察下上述的三个网络，Pnet经过几层的conv，最终输出1x1x2、4、10，注意这个Pnet输出和后面Rnet、Onet有点不同。区别在于Pnet是输出的map，比如图片是h*w大小，那么经过这个网络就会输出也大体是图片的1/2（这个不一定，看你使用的conv和pooling，大体会计算得出一个feature map 大小为：(h_new, w_new, 16）)。而Rnet和Onet输出的是每个阶段（num_of_left, 16)个candidate。这也就解释了为什么Pnet的输出是方块，而Rnet和Onet输出是长条形。

##  训练数据

理解完这点后，我们继续下去，探索下每个网络所使用的训练图片特点

先讲一个统一的东西，每个网络的输入我们会有4种训练数据输入：

\* Positive face数据：图片左上右下坐标和label的IOU>0.65的图片

\* part face数据：图片左上右下坐标和label的0.65>IOU>0.4的图片

\* negative face 数据：图片左上右下坐标和lable的IOU<0.3的图片

\* landmark face数据：图片带有landmark label的图片

为什么要这么分？因为我们叫做MTCNN，Multi-task，所以以上的图片的真正用途是如下：

为什么要这么分？因为我们叫做MTCNN，Multi-task，所以以上的图片的真正用途是如下：

\- 网络做人脸分类的时候，使用postives 和negatives的图片来做，为什么只用这两种？因为这两种数据分得开，中间隔着个part face+0.1 IOU的距离，容易使模型收敛；

\- 网络做人脸bbox的偏移量回归的时候，使用positives 和parts的数据，为什么不用neg数据？论文里面没提，个人认为是因为neg的数据几乎没有人脸，用这个数据来训练人脸框offset的回归挺不靠谱的，相反pos和part的数据里面人脸部分比较大，用来做回归，网络还能够看到鼻子、眼睛、耳朵啥的来进行乖乖的回归；

\- 网络做人脸landmark 回归的时候，就只使用landmark face数据了。

  Pnet使用的是12*12大小的图片，这个图片怎么得到的呢？嗯，很简单，去WIDER和CelebA随机截取，这个时候大家会问，随机截取怎么截取？就是字面上的意思，不过有点点技巧。首先，如果真的随机截取的话，如果图片里面人头只有一个，很多都会截取到非pos甚至非part的图片，所以为了得到足够多的pos、part数据，真正的随机截取是基于图片实际label进行上下左右微调来截取，进而保障pos、part数据的足够。举个例子，上面老爷爷照片，我们是有label的，他告诉我们人脸框的位置假设是左上（x1， y1）右下（x2， y2），那么我们截取的话就可以基于这两个坐标做稍微的左右上下微调如x1往左0.1个w（=x2-x1），y1往下0.15个h（y2-y1）等等，这样就能够确保你得到足够的pos和part数据。最终你会得到很多还没有resize的图片，长宽不一，这个时候你把他们resize为12*12大小的图片即可。这个时候我们X有了，但是我们其实还缺少label，这些12*12的图片的label应该怎么标？

  其实是这样的，首先根据图片的种类，pos为1，part为-1， neg为0，landmark为-2，这个用来训练分类。

  bbox的label应该怎么标？还记得我们一直提到的偏移量（offset）吗？我们就是根据offset来标的，好希望有黑板啊，我还得找笔来画个图。啊摔，暂时找不到，明天再补图吧，先说：假设label为：左上（xlu1， ylu2）右下（xrd，yrd2）， 截图图片实际在原图的绝对坐标为（x1， y1）（x2， y2），那么我们左上x1的offset就是定义为：(x1-xlu1)/（x2-x1）是个归一化的数据，同理可以得到左上y1右下x2、右下y2的offset。简单点的理解就是左上、右下的x和y坐标基于截图宽和高往上和往下偏移了多少倍？0.1？-0.1还是0.3，-0.2。



  ![img](https://pic1.zhimg.com/80/v2-4056469d28bd692d20173a389ebfb608_720w.jpg)bbox的label怎么标



  为啥要用归一化数据？可以抗resize的干扰，原图上计算好了之后，resize成12*12就还是归一化的量。由于Pnet输入需要是12*12的，所以截图还需要resize，但是offset的label就不用做变换啦。

  下面讲讲landmark的label怎么标，同理也是offset，相较于上述根据左上右下每个x、y的偏移量，这次landmark是仅基于左上的坐标的偏移量。来张图最直观：



  ![img](https://pic4.zhimg.com/80/v2-dfd15ba1cfa55ef44555f64aaad8de6f_720w.jpg)landmark的lable如何标



  OK，图片随机截取和label的标记已经讲完了，那么结果是怎么样的呢？你们应该很关注：



  ![img](https://pic4.zhimg.com/80/v2-870b16041f5beeb2dca3abcf9123593b_720w.jpg)12*12的图片



  ![img](https://pic4.zhimg.com/80/v2-7383a9f1d1531b05a1f087d43aaff44f_720w.jpg)neg的标记，除了0后面没有别的了



  ![img](https://pic1.zhimg.com/80/v2-84a860be4dc3af2eb537b7ad6ccc4b08_720w.jpg)positive的标记，1 和后面的偏移量



  ![img](https://pic4.zhimg.com/80/v2-081ccf3063e5cf4553d3280710aa249f_720w.jpg)part的标记，-1和后面的偏移量



  ![img](https://pic4.zhimg.com/80/v2-92946da9eba8df3a6a7d6eadffd319c7_720w.png)landmark face的标记，-2和后面5个点的偏移量

  OK，pnet训练数据怎么造已经讲清楚了，接下来讲讲Rnet和Onet，注意Rnet和Onet的数据就不是随机crop出来了，而是分别由前面网络的输出得到的训练数据。

  Pnet在前述数据的情况下进行训练并完成训练，我们将所有的WIDER数据和CelebA数据输入到Pnet，会得到很多的候选，去原图截图，计算截图和label的IOU，按照上述四种数据的分类标准分别区分开，同时label标注方法和上述标注方法一致。我们经过Pnet就可以得到Rnet所需的24*24大小的训练数据了。我们继续训练好Rnet。

  Onet的输入是WIDER数据集和CelebA数据集经过Pnet、Rnet后得到的在原图上截取的图片，根据IOU的不同进行分类，并resize为48*48。这样子我们再把Onet给训练好。

## 损失函数

  讲完上述训练数据的构造，我们再来提提损失函数：

  每个网络都会有三个损失函数：分类使用的是交叉熵损失函数，bbox回归使用的是平方差损失函数、landmark回归使用的也是平方差损失函数。这些都比较好理解，需要注意的地方在于下面这幅图：

  ![img](https://pic1.zhimg.com/80/v2-75ceb768d82d2e94513d9ba7c5eddfe0_720w.png)



  这个损失函数表达的意思是：

  \- alpha是表示不同网络结构det、box、landmark的损失函数的权重不一，由于Onet要输出landmark，所以Onet的landmark损失权重会比前面两个网络要大。

  \- Beta的意思是数据type指示器（indicator），简单的理解就是一个batch的图片feed到网络后，如果是neg图片，那么他只贡献det loss，Beta_det=1其余两个为0，如果是pos 图片，则他不仅贡献det loss，还贡献box loss（即bbox回归loss）beta_det=1和beta_box=1,beta_landmark=0，如果图片是part face则它仅贡献于box loss, beta_box=1,其余两个beta为0，如果图片是landmark face，则它仅贡献于landmark loss，beta_landmark=1，其余两个为0。

  \- **Hard Sample mining：只对分类损失进行hard sample mining，**具体意思是在一个batch里面的图片数据，只取分类损失（det loss）的前70%的训练数据（这个比例是提前设好的）backprop回去。其余两类损失不做这样的hard sample mining，原因在于回归问题再微小的nudge修正都是有用的，但是二分类就未必了。

## 填一个坑：

  推阶段：Pnet输出的bbox偏移量和landmark的偏移量是基于什么图片的偏移量？

  答：假设（h, w, 3）的图片经过Pnet变成了（h/2, w/2, 16），那么feature map内所有像素映射回原图的位置，并向右向下推12个像素得到的box就是对应的截图图片。如果你们有看过Faster RCNN的文章的话，这个思想和RPN是基本一致的，这里的滑动窗口是12*12.

# 总结



  花了好一些时间来写这篇MTCNN的详解文章，感觉还不是很完善，希望能够与读者朋友们的交流下进行完善。

  总结下MTCNN的流程：图片经过Pnet，会得到feature map，通过分类、NMS筛选掉大部分假的候选；然后剩余候选去原图crop图片输入Rnet，再对Rnet的输出筛选掉False、NMS去掉众多的候选；剩余候选再去原图crop出图片再输入到Onet，这个时候就能够输出准确的bbox、landmark坐标了。

  这是一个coarse to fine 的过程。