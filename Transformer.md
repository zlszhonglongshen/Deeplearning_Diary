# Transformer学习详解

## 前言

Transformer本质是一种seq2seq结构，那么它一定也有encoder和decoder部分，只是这两个部分不同于以往模型采用的RNN结构，Transformer聪明地开创了另一种新的结构。

## Transformer结构

以中英文翻译为例，可以看看它的整体结构

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004144226675.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center)

可以看到Transformer由6个encoder堆叠而成，6个decoder堆叠而成。

我们再看看每个encoder和decoder具体结构是什么样子。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020100414502512.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center)

其中，Nx表示有6个这样的结构。

可以看到每一个encoder有两个子层：多头注意力层和全连接前馈神经网络层。子层的连接使用了LayerNorm和残差连接，可以避免梯度消失和爆炸。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004152303577.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center)

每个decoder有3个子层：带masked的多头自注意力层，多头自注意力层和全连接层神经网络层。

**Transformer的大体流程是：**

编码器（由6个encoder组成）对一个输入序列的embedding生成一个顺序编码的输出Z，而解码器（由6个decoder组成）根据编码器的输出Z，一次生成一个输出y1,Z和y1同时作为输入生成下一步的y2,最终得到生成序列Y。

## 详细的工作流程

**第一步：**获取输入句子的每一个**单词**的表示向量X，X由单词的embedding和单词**位置**的embedding相加得到。

<img src="https://img-blog.csdnimg.cn/20201004145956699.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:80%;" />

**第二步：**将得到的单词embedding矩阵（如上图所示：我有一只猫，每一行是一个单词的词向量表示x）传入编码器中，经过6个encoder block后可以得到句子所有单词的编码信息矩阵Z，如下图。单词向量矩阵用X（nxd)表示，n是句子中单词个数，d是表示向量的维度（论文中d=512）。每一个encoder block输出的矩阵维度和输出完全一直。

<img src="https://img-blog.csdnimg.cn/20201004150249669.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

**第三步：**将encoder输出的编码信息矩阵Z传递到解码器中，编码器依次会根据当前翻译过的1~i个单词，翻译下一个单词i+1，如下图所示。在使用的过程中，翻译到单词i+1的时候需要通过MASK（掩盖）操作遮盖住i+1之后的单词。

<img src="https://img-blog.csdnimg.cn/20201004150524561.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

上图 解码器 接收了 编码器的信息矩阵Z，然后先传入一个翻译开始符 “< Begin >”，预测第一个单词 **“I”**；然后传入翻译开始符 “< Begin >” 和单词 “I”，预测下一个单词 “have”，以此类推。

这是Transformer使用时候的大致流程，接下来是里面各个部分的细节。

## Transformer细节信息

### Transformer的输入

Transformer中单词的输入表示x由单词embedding和位置embedding相加得到。

<img src="https://img-blog.csdnimg.cn/20201004150820490.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

**单词Embedding**

单词的Embedding有很多中方式可以获取，例如可以采用word2vec、Glove等算法预训练得到，也可以在Transformer中训练得到。

**位置Embedding**

Transformer中除了单词的Embedding，还需要使用位置的Embedding表示单词出现在句子中的位置。因为Transformer不采用RNN的结构，而是使用了句子全局信息，这样就不能保证单词的顺序信息，而这部分对于语言模型来说非常重要。所以Transformer中使用位置Embedding保存单词在序列中的相对或绝对位置。

位置Embedding用PE表示，PE的维度和单词Embedding是一样的。

可以使用某种公式计算得到，公式如下

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004151114254.png#pic_center)

其中，pos 表示单词在句子中的位置，d 表示 PE 的维度 (与词 Embedding 一样)，i 是每个维度，2i 表示偶数的维度，2i+1 表示奇数维度 (即 2i≤d, 2i+1≤d)。记住 “奇数位：cos，偶数位：sin”

将单词的词Embedding和位置Embedding相加，就可以得到单词的表示向量x，x就是Transformer的输入。

### multi-head self-attention

<img src="https://img-blog.csdnimg.cn/20201004151426529.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

上图是论文中Transformer的内部结构图，左侧为编码器，右侧为解码器，红色圈中的部分为Multi-Head self-Attention,是由多个self-Attention组成。

**self-attention**

* 如图，self-attention计算过程

  <img src="https://img-blog.csdnimg.cn/20201004151840395.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

* 计算公式

  <img src="https://img-blog.csdnimg.cn/2020100415313098.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

  在计算的时候需要用到矩阵Q(查询）K（键值），V（值）。在实际中运用时，self-Attention接收的是输入（单词的表示向量x组成的矩阵X）或者上一个Encoder block**输出**

  而Q，K，V正是通过self-Attention的输入X进行**线性变换**得到的。如图（批训练的话，向量就变成矩阵了，也可以说成 键矩阵，查询矩阵，值矩阵）

  **MarMul** : 矩阵乘法 即 QK^T
  **scale**: 尺度缩放 即 除以 sqrt(d_k)
  **mask**: 在Encoder 中没有，只在Decoder中有
  **softmax**：归一化

  <img src="https://img-blog.csdnimg.cn/20201004152422271.png#pic_center" alt="在这里插入图片描述" style="zoom:100%;" />

* **Q，K，V的计算**

  Self-Attention 的输入用矩阵 X 进行表示，则可以使用线性变阵矩阵 WQ, WK, WV 计算得到 Q, K, V。计算如下图所示，注意 X, Q, K, V 的每一行都表示一个单词。

  <img src="https://img-blog.csdnimg.cn/20201004152716364.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

  这里要注意：健向量，查询向量，值向量维度一般比Embedding词向量低，在原论文中是输入单词Embedding维度的 1/8，（思考：为什么维度要变成Embedding词向量1/ 8，因为论文刚好建立了 8 个 自注意力机制，每个自注意力机制的 Q, K, V 维度就是（512 / 8), 最后再拼接这 8 个自注意了机制的结果，维度也能回到 512）
  

* **self-attention值的计算**

  得到Q，K，V之后

  <img src="https://img-blog.csdnimg.cn/20201004153957382.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

  1） Q 乘以 K 的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 **attention 强度**。下图为 Q 乘以 K 的转置，1234 表示的是句子中的单词。

  <img src="https://img-blog.csdnimg.cn/20201004154013414.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

  2）得到 QK^T 之后，使用 Softmax 计算每一个单词对于其他单词的 attention 系数，公式中的 Softmax 是对矩阵的**每一行**进行 Softmax，即每一行的和都变为 1。

  <img src="https://img-blog.csdnimg.cn/20201004154110241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

3) 得到 Softmax 矩阵之后可以和 V 相乘，得到最终的编码信息输出 Z。

   <img src="https://img-blog.csdnimg.cn/20201004154143315.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

   4）上图中 Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出 Z1 等于所有单词 i 的值 Vi 根据 attention 系数的比例加在一起得到，如下图所示：

   <img src="https://img-blog.csdnimg.cn/2020100415431494.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

   

* **Multi-Head**

  在上一步，我们已经知道怎么通过 Self-Attention 计算得到输出矩阵 Z，而 Multi-Head Self Attention 是由多个 Self-Attention 组合形成的，下图是论文中 Multi-Head Attention 的结构图。

  <img src="https://img-blog.csdnimg.cn/20201004155135647.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

  从上图可以看到 Multi-Head Attention 包含多个 Self-Attention 层，首先将输入 X 分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵 Z。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵 （Z1,Z2, Z3, Z4, Z5,Z6,Z7,Z8）。

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004155412944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center)

  可以先看上图：X是输入的向量表示，维度是[batch_size,seq_len,d]

  Batch_size:一个批大小

  seq_len:句子长度

  d：词向量维度，论中中是512

  **如果只有一个头**：那么 健向量的维度就是[ batch_size, seq_len, 512], 但现在我们有8个头，之前也提到将 Q,K,V维度降到原来的 1/8 ，就是为了这 8 个头
  所以**8头**： 健向量的维度就是[ batch_size, seq_len, 8, 64]

  **流程如下**

  <img src="https://img-blog.csdnimg.cn/20201004160144143.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

  得到 8 个输出矩阵 Z1 到 Z8 之后，Multi-Head Attention 将它们拼接在一起 (Concat)，然后传入一个 Linear 层，得到 Multi-Head Attention 最终的输出 Z。

  <img src="https://img-blog.csdnimg.cn/20201004160251273.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

  可以看到Multi-Head attention输出的矩阵Z与其输入的矩阵X的维度是一样的。

  

  ### Add&Norm

  <img src="https://img-blog.csdnimg.cn/20201004160413796.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

  上图红色部分是Transformer的encoder block结构，可以看到的是由Multi-head Attention,Add & Norm, Feed Forward, Add & Norm 组成的。刚刚已经了解了 Multi-Head Attention 的计算过程，现在了解一下 Add & Norm 和 Feed Forward 部分。

  

* add&norm层由add和norm两部分组成，计算公式如下

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004160502673.png#pic_center)

  Add 指 X+MultiHeadAttention(X)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分，在 ResNet 中经常用到。

* 残差连接

  <img src="https://img-blog.csdnimg.cn/2020100416073869.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

  残差连接简单说就是：计算几层输出之后**再**把x加进来。都说残差网络有效的解决了梯度消失的问题，解决了网络退化的问题。下面是简化图：

  <img src="https://img-blog.csdnimg.cn/20201004161146865.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

* Norm 指 Layer Normalization，通常用于 RNN 结构，Layer Normalization 会将每一层神经元的输入都转成**均值方差**都一样的，这样可以加快收敛。



### Feed Forward

Feed forward层比较简单，是一个两层的全连接层，第一层的激活函数为Relu，第二层不使用激活函数，对应当公式如下。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004161236282.png#pic_center)

X是输入，Feed Forward最终得到的输出矩阵的维度与X一直。

## 组成Encoder

* 通过上面描述的 Multi-Head Attention, Feed Forward, Add & Norm 就可以构造出一个 Encoder block，Encoder block 接收输入矩阵 X(n×d)，并输出一个矩阵 O(n×d)。通过多个 Encoder block 叠加就可以组成 编码器。
*  第一个 Encoder block 的输入为句子单词的**表示向量矩阵**，后续 Encoder block 的**输入**是前一个 Encoder block 的**输出**，最后一个 Encoder block 输出的矩阵就是 编码信息矩阵Z，这一矩阵后续会用到 Decoder 中。

### Decoder结构

<img src="https://img-blog.csdnimg.cn/20201004161559301.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" />

上面红色部分为Transformer的Decoder block结构，与Encoder block相似，但是存在一些区别：

* 包含两个Multi-Head attention层。
* 第一个multi-head attention层采用了masked操作
* 第二个multi-head Attention层的K，V矩阵使用encoder编码信息矩阵Z进行计算（encoder是使用输入X或前一个encoder的输出），而Q使用上一个decoder的输出进行计算。（如果是第一个decoder则使用输入矩阵X进行计算）
* 最后有一个softmax层计算下一个翻译单词的概率

### 第一个mask multi-head Attention

decoder block的第一个multi-head attention采用了masked操作，因为在翻译的过程中是顺序翻译的，即翻译完第i个单词，才可以翻译第i+1个单词。

通过**masked**操作可以防止第i个单词知道i+1个单词之后的信息。

下面以 “我有一只猫” 翻译成 “I have a cat” 为例，了解一下 Masked 操作。

**训练过程**

Decoder在训练的过程中使用Teacher Forcing进行训练，即将**正确**的单词序列 ( < Begin > I have a cat) 和对应**输出** (I have a cat < end >) 传递到 Decoder中。那么在预测第 **i** 个输出时，就要将第 **i+1** 之后的单词掩盖住。

下面用 0 1 2 3 4 5 分别表示 “< Begin > I have a cat < end >”。

第一步：是 Decoder 的输入矩阵和 Mask 矩阵，输入矩阵包含 “< Begin > I have a cat” (0, 1, 2, 3, 4) 五个单词的表示向量，Mask 是一个 5×5 的矩阵。在 Mask 可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息


![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004162611484.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center)

**第二步**：接下来的操作和之前的 Self-Attention 一样，通过输入矩阵 X 计算得到 Q, K, V 矩阵。然后计算 Q 和 KT 的乘积 QK^T。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004162702370.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center)

**第三步**：在得到 QK^T 之后需要进行 Softmax，计算 **attention score**, **但**我们在 **Softmax 之前**需要使用 **Mask** 矩阵遮挡住每一个单词之后的信息，遮挡操作如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004162834537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center)

得到 Mask QKT 之后在 Mask QK^T 上进行 Softmax，每一行的和都为 1。但是单词 0 在单词 1, 2, 3, 4 上的 attention score 都为 0, 因为它们被遮盖住了，不需要关注。
第四步：使用 Mask QK^T 与矩阵 V 相乘，得到输出 Z，则单词 1 的输出向量 Z1 是包含单词 0, 单词1 信息的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004162927418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjQyNTY5Mg==,size_16,color_FFFFFF,t_70#pic_center)

**第五步**：通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵 Zi，然后和 Encoder 类似，通过 **多头** 拼接多个输出Zi 然后计算得到第一个 Multi-Head Attention 的输出 Z，Z与输入 X 维度一样。

### 第二个 Multi-Head Attention
Decoder block 第二个 Multi-Head Attention 变化不大， 主要的区别在于其中 Self-Attention 的 K, V 矩阵不是使用 上一个 Decoder block 的输出计算的，而是使用 Encoder 的编码信息矩阵 Z 计算的。
根据 Encoder 的输出 Z 计算得到 K, V，根据上一个 Decoder block 的输出 计算 Q (如果是第一个 Decoder block 则使用输入矩阵 X 进行计算)，后续的计算方法与之前描述的一致。
这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 Mask)。
这就是 Decoder block 的定义，与 Encoder 一样，Decoder 是由多个 Decoder block 组合而成
## Transformer总结
Transformer 与 RNN 不同，可以比较好地并行训练。
Transformer 本身是不能利用单词的顺序信息的，因此需要在输入中添加位置 Embedding，否则 Transformer 就是一个词袋模型了。
Transformer 的重点是 Self-Attention 结构，其中用到的 Q, K, V 矩阵通过输入进行线性变换得到。
Transformer 中 Multi-Head Attention 中有多个 Self-Attention，可以捕获单词之间多种维度上的相关系数 attention score。


## 参考文献
https://blog.csdn.net/weixin_46425692/article/details/108918850