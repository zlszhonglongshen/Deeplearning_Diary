# LSTM

```
记住了结论：第一步决定什么被遗忘，第二步决定什么被记住，第三步更新状态。
```



>1. 什么是LSTM？
>2. LSTM核心思想是什么？
>3. 如何理解LSTM？

### LSTM网络

long short term，即我们所称呼的LSTM，是为了解决长期以来问题而设计出来的。所有的RNN都具有一种重复神经网络的链式形式。在标准RNN中，这个重复的结构模块只有一个非常简单的结构，例如一个tanh层。

![img](https://www.aboutyun.com/data/attachment/forum/201904/16/180607acxjjc4vevjj4hby.jpg)

LSTM同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于单一神经网络层，这里是有4个，以一种非常特殊的方式进行交互。

![img](https://www.aboutyun.com/data/attachment/forum/201904/16/180621utrbxb8btrbtwxrw.jpg)

不必担心这里的细节。我们会一步一步的剖析LSTM解析图。现在，我们先来熟悉一下图中使用的各种元素的图标。

![img](https://www.aboutyun.com/data/attachment/forum/201904/16/180637g44lrlzitrhh9yez.jpg)

在上面的图例中，每一条黑线传输着一整个向量，从一个节点的输出到其他节点的输入。粉色的圈代表pointwise的操作，诸如向量的和，而黄色的矩阵就是学习到的神经网络层。合在一起的线表示向量的连接，分开的线表示内容被复制，然后分发到不同的位置。

### LSTM核心思想

LSTM的关键在于细胞的状态和穿过细胞的那条水平线。

细胞状态类似于传送带。直接在整个连上运行，只有一些少量的线性交互。信息在上面流转保持不变会很容易。

![img](https://www.aboutyun.com/data/attachment/forum/201904/16/180707i29j5ec7g7z2392j.jpg)

若只有上面的那条水平线是没办法实现添加或者删除信息的。而是通过一种叫做门（gates)的结构来实现的。

门 是可以实现选择性的让信息通过，主要是通过一个sigmoid的神经层和一个逐点相乘的操作来实现的。

![img](https://www.aboutyun.com/data/attachment/forum/201904/16/180723i598fzelvexhwlbb.jpg)

sigmoid层输出（是一个向量）的每个元素都是一个在0和1之间的实数，表示让对应信息通过的权重（或者占比）。比如，0表示“不让任何信息通过”，1表示“让所有信息通过”。

LSTM通过三个这样的结构来实现信息的保护和控制。这三个门分别是输入门、遗忘门和输出门。

### 逐步理解LSTM

现在我们就开始通过三个门逐步了解LSTM的原理

#### 遗忘门

LSTM的第一步就是决定什么信息应该被神经元遗忘。这是一个被称为“遗忘门层”的Sigmod层组成的。它输入 ht−1和xt,然后在Ct−1 的每个神经元状态输出0~1之间的数字。“1”表示“完全保留这个”，“0”表示“完全遗忘这个”。

让我们再次回到那个尝试去根据之前的语法与预测下一个第N次的语言模型。在这个问题中，神经元转改或许包括当前主语中的性别信息，所以可以使用正确的代词。当我们看到一个新的主语，我们回去遗忘之前的性别信息。

![img](https://pic4.zhimg.com/80/v2-4035f737ddf1b26add27f4f69ccc1483_1440w.png)

下一步就是我们决定我们要在神经元细胞中保存什么信息，这包括两个部分。首先，一个被称为“输入门层”的sigmoid层决定我们要更新的数值。然后，一个tanh层生成一个新的候选数值。Ct˜,它会被增加到神经元状态中。在下一步中中，我们会组合这两步去生成一个更新状态值。
在那个语言模型例子中，我们想给神经元状态增加新的主语的性别，替换我们将要遗忘的旧的主语。

![img](https://pic1.zhimg.com/80/v2-5a460ea6f112de7332ca9584300c6e9c_1440w.png)

是时候去更新旧的神经元状态Ct−1到新的神经元状态Ct了。之前的步骤已经决定要做什么，下一步我们就去做。
我们给旧的状态乘以一个ft,遗忘掉我们之前决定要遗忘的信息，然后我们增加it∗Ct˜。这是新的候选值，是由我们想多大程度上更新每个状态的值来度量的。
在语言模型中，就像上面描述的，这是我们实际上要丢弃之前主语的性别信息，增加新的主语的性别信息的地方。

![img](https://pic3.zhimg.com/80/v2-830f08d8fff0c45ff6350c8c473b50ba_1440w.png)

最后，我们要决定要输出什么。这个输出是建立在我们的神经元状态的基础上的，但是有一个滤波器。首先，我们使用Sigmod层决定哪一部分的神经元状态需要被输出；然后我们让神经元状态经过tanh（让输出值变为-1~1之间）层并且乘上Sigmod门限的输出，我们只输出我们想要输出的。
对于那个语言模型的例子，当我们看到一个主语的时候，或许我们想输出相关动词的信息，因为动词是紧跟在主语之后的。例如，它或许要输出主语是单数还是复数的，然后我们就知道主语联结的动词的语态了。

![img](https://pic3.zhimg.com/80/v2-c3c06077f1a4e436ac442e0623ac284e_1440w.png)

### 代码

```
# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x

if __name__ == '__main__':
    # create database
    data_len = 200
    t = np.linspace(0, 12*np.pi, data_len)
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    dataset = np.zeros((data_len, 2))
    dataset[:,0] = sin_t
    dataset[:,1] = cos_t
    dataset = dataset.astype('float32')

    # plot part of the original dataset
    plt.figure()
    plt.plot(t[0:60], dataset[0:60,0], label='sin(t)')
    plt.plot(t[0:60], dataset[0:60,1], label = 'cos(t)')
    plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5') # t = 2.5
    plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8') # t = 6.8
    plt.xlabel('t')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('sin(t) and cos(t)')
    plt.legend(loc='upper right')

    # choose dataset for training and testing
    train_data_ratio = 0.5 # Choose 80% of the data for testing
    train_data_len = int(data_len*train_data_ratio)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    t_for_training = t[:train_data_len]

    # test_x = train_x
    # test_y = train_y
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM) # set batch size to 5
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM) # set batch size to 5
 
    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)
    # test_x_tensor = torch.from_numpy(test_x)
 
    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=1) # 16 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
 
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
 
    max_epochs = 10000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        loss = loss_function(output, train_y_tensor)
 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch+1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))
 
    # prediction on training dataset
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files
 
    # ----------------- test -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval() # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 5, INPUT_FEATURES_NUM) # set batch size to 5, the same value with the training set
    test_x_tensor = torch.from_numpy(test_x_tensor)
 
    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
 
    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_x, 'g', label='sin_trn')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_cos_trn')

    plt.plot(t_for_testing, test_x, 'c', label='sin_tst')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_cos_tst')

    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line') # separation line

    plt.xlabel('t')
    plt.ylabel('sin(t) and cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size = 15, alpha = 1.0)
    plt.text(20, 2, "test", size = 15, alpha = 1.0)

    plt.show()
```



### 参考
https://zhuanlan.zhihu.com/p/28054589

https://zhuanlan.zhihu.com/p/24018768

https://www.aboutyun.com/forum.php?mod=viewthread&tid=27020