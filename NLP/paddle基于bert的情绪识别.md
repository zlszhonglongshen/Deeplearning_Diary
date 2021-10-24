# paddle基于bert的情绪识别

## 文件结构

```
* bert-paddle 存放预训练模型路径
   * vocab.txt 字典文件，该字典为大小为21128。
   * model_config.json 模型配置文件。
   * model_state.pdparams 模型参数文件。
* data_dir 存放数据的文件夹
	* usual_train.txt 原始训练集文件。
   * usual_eval_labeled.txt 原始测试集文件。
* data_helper.py 数据预处理文件，将数据进行简单的格式转换。
* data_set.py 数据类文件，定义模型所需的数据类，方便模型训练使用。
* model.py 情绪识别模型文件，主要对transformers包中BertPretrainedModel的重写。
* train.py 情绪识别模型训练文件。
* predict.py 根据训练好的模型，进行情绪预测，并且包含了动态图、onnx和静态图的时间评测。
* requirements.txt 环境配置文件，按照一些额外的包。
```

## 数据集

数据集来自SMP2020微博情绪分类评测比赛中通用微博数据集。按照其蕴含的情绪分为以下六个类别之一：积极、愤怒、悲伤、恐惧、惊奇和无情绪。 SMP2020微博情绪分类评测比赛链接：https://smp2020ewect.github.io/

| 情绪   | 文本                                                         |
| ------ | ------------------------------------------------------------ |
| 积极   | 哥，你猜猜看和喜欢的人一起做公益是什么感觉呢。我们的项目已经进入一个新阶段了，现在特别有成就感。加油加油。 |
| 愤怒   | 每个月都有特别气愤的时候。，多少个瞬间想甩手不干了，杂七杂八，当我是什么。 |
| 悲伤   | 回忆起老爸的点点滴滴，心痛…为什么.接受不了                   |
| 恐惧   | 明明是一篇言情小说，看完之后为什么会恐怖的睡不着呢，越想越害怕[吃驚] |
| 惊奇   | 我竟然不知道kkw是丑女无敌里的那个                            |
| 无情绪 | 我们做不到选择缘分，却可以珍惜缘分。                         |

## 数据预处理

数据预处理代码，主要是将其原始数据格式进行转换，查看数据集中各个类别的占比。其实，正常项目，还可以增加一些数据清洗的工作（本项目省略了数据清洗的部分）。

```
import json
def data_proprecess(path,save_path):
    '''
    将原始数据格式转换成模型所需格式数据，并统计各标签数据的数量
    '''
    input = open(save_path,"w",encoding="utf-8")
    data_number = {}
    with open(path,"r",encoding="utf-8") as f:
        #加载原始数据
        data = json.load(f)
        #对原始数据进行遍历
        for i,line in enumerate(data):
            sample = {"text":line["content"],"label":line["label"]}
            #如果标签在data_number中，直接对其value进行加1操作；如果不在，则将标签加入的data_number中，value设为1。
            if line["label"] not in data_number:
                data_number[line['label']]= 1
            else:
                data_number[line["label"]] += 1
                
            #将每一个文本和对应的标签，写入到保存文件中
            input.write(json.dumps(sample,ensure_ascii=False)+"\n")
    print("data_number:",data_number)
   
```

```
train_path = "./data/usual_train.txt"
save_train_path = "./data/train.json"
data_proprecess(train_path, save_train_path)

test_path = "./data/usual_eval_labeled.txt"
save_test_path = "./data/test.json"
data_proprecess(test_path, save_test_path)
```

```
data_number: {'angry': 8344, 'happy': 5379, 'neutral': 5749, 'surprise': 2086, 'sad': 4990, 'fear': 1220}
data_number: {'angry': 586, 'happy': 391, 'sad': 346, 'neutral': 420, 'fear': 87, 'surprise': 170}
```

## 数据类实现

数据类的作用是将文本数据转换成模型可以使用的索引数据，并预先存储下来。避免模型每训练一步，都进行无效的数据转换操作。

```
#导入程序包
from paddle.io import Dataset
from paddlenlp.data import Pad,Stack,Dict
import paddle
import json
import os 
import logging

logger = logging.getLogger(__name__)
```

定义模型所需的SentimentAnalysisDataSet数据类，继承paddle.io.Dataset类，包含__init__函数、load_data函数、convert_featrue函数、__len__函数、以及__getitem__函数

```
class SentimentDataSet(Dataset):
    def __init__(self,tokenizer,max_len,data_dir,data_set_name,path_file=None,is_overwrite=False):
        """
        模型所需的数据类，继承paddle.io.Dataset类
        Args:
            tokenizer:分词器
            max_len:文本最大长度
            data_dir:保存缓存数据路径
            data_set_name:数据集名字
            path_file:原始数据文件路径
            is_overwrite:是否对缓存文件进行重写
        """
        super(SentimentDataSet,self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        #6种标签字典
        self.label2id = {'angry': 0, 'happy': 1, 'neutral': 2, 'surprise': 3, 'sad': 4, 'fear': 5}
        self.id2label = {0: "angry", 1: "happy", 2: "neutral", 3: "surprise", 4: "sad", 5: "fear"}
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        #判断如果存在缓存文件，则直接对其进行加载
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{},直接加载".format(cached_feature_file))
            self.data_set = paddle.load(cached_feature_file)["data_set"]
        else:
            #如果不存在缓存文件，则调用load_data函数，进行数据预处理，再将其保存成缓存文件。
            logger.info("不存在缓存文件{},进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            paddle.save({"data_set": self.data_set}, cached_feature_file)
            
            
    def load_data(self,path_file):
        """
        对原始数据中每一条数据预处理操作，将文本转换成模型可以用的id索引形式
        """
        data_set = []
        with open(path_file,"r",encoding="utf-8") as f:
            for i,line in enumerate(f):
                #加载每一条数据
                sample = json.loads(line.strip())
                #调用convert_feature函数，对单条数据进行文本转换成操作
                input_ids,attention_mask,label = self.convert_feature(sample)
                sample["input_ids"] = input_ids
                sample["attention_mask"] = attention_mask
                sample["label"] = label
                #将数据存放到data_set中
                data_set.append(sample)
        return data_set
                
    
    def convert_feature(self,sample):
        '''
        将单个样本转换成模型可用的id索引形式
        '''
        #获取标签索引
        label = self.label2id[sample["label"]]
        #将文本进行tokenize
        tokens = self.tokenizer.tokenize(sample["text"])
        #进行长度判断，若长于最大长度，则进行截断
        if len(tokens)>self.max_len-2:
            tokens = tokens[:self.max_len-2]
        #将其头尾加上[CLS]和[SEP]
        tokens = ["[CLS]"]+tokens+["SEP"]
        #将token转换成id
        inputs_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #获取模型所需的attention_mask,大小和input_ids一致
        attention_mask = [1]*len(inputs_ids)
        assert len(inputs_ids)==len(attention_mask)
        return inputs_ids,attention_mask,label
    
    def __len__(self):
        """获取数据总长度"""
        return len(self.data_set)

    def __getitem__(self, idx):
        """按照索引，获取data_set中的指定数据"""
        instance = self.data_set[idx]
        return instance
```

在模型训练时，对batch数据进行tensor转换函数，定义DataLoader所需的collate_fun函数，将数据处理成tensor形式。

```
def collate_func_sentiment_analysis(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据

    Returns:

    """
    # 获取batch数据的大小
    batch_size = len(batch_data)
    # 如果batch_size为0，则返回一个空字典
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, labels_list = [], [], []
    # 遍历batch数据，将每一个数据，转换成tensor的形式
    for instance in batch_data:
        input_ids_temp = instance["input_ids"]
        attention_mask_temp = instance["attention_mask"]
        labels_temp = instance["label"]
        input_ids_list.append(paddle.to_tensor(input_ids_temp, dtype="int64"))
        attention_mask_list.append(paddle.to_tensor(attention_mask_temp, dtype="int64"))
        labels_list.append(labels_temp)
        
    #对一个batch内的数据，进行padding
    return {"input_ids": Pad(pad_val=0, axis=0)(input_ids_list),
            "attention_mask": Pad(pad_val=0, axis=0)(attention_mask_list),
            "label": Stack(dtype="int64")(labels_list)}
```

## 模型代码实现

模型部分，主要使用PaddleNLP的transformers的BertPretrainedModel类实现模型代码。

```
import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertPretrainedModel
import paddle.nn.functional as F
```

### 模型函数，主要继承Bert模型

```
class Model(BertPretrainedModel):
    base_model_prefix = "bert"
    def __init__(self,bert,number_label=3):
        """
        主要继承paddlenlp.transformers.BertPretrainedModel类
        """
        super(Model,self).__init__()
        self.bert = bert
        self.classifier = nn.layer.Linear(self.bert.config["hidden_size"], number_label)
        self.loss_fct = nn.CrossEntropyLoss(soft_label=False,axis=1)
        
    
    def forward(self,input_ids,attention_mask,label=None):
        # 将attention_mask进行维度变换，从2维变成4维。paddlenlp.transformers的实现与torch或tf不一样，不会自动进行维度扩充。
        attention_mask = paddle.unsqueeze(attention_mask,axis=[1,2])
        #获取CLS向量pooled_output
        pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask)[1]
        #对pooled_output进行全链接，映射到number_label上
        logits = self.classifier(pooled_output)
        #使用softmax 获取每个标签类别的概率
        probs = F.softmax(logits,axis=1)
        #获取标签类别概率最大的标签
        pred_label = paddle.argmax(logits,axis=-1)
        outputs = (pred_label,probs)
        #如果label不是None，则使用CE求解loss
        if label is not None:
            loss = self.loss_fct(logits,label)
            outputs = (loss,)+outputs
        return outputs
```

## 模型训练

```
import paddle
import os
import random
import numpy as np
import argparse
import logging
import json
from paddlenlp.transformers import BertTokenizer
from paddle.io import DataLoader, SequenceSampler, RandomSampler, BatchSampler
from paddlenlp.data import Pad, Stack, Dict
from paddlenlp.ops.optimizer import AdamW
from paddlenlp.transformers import LinearDecayWithWarmup
from tqdm import tqdm, trange
from sklearn.metrics import classification_report, f1_score, accuracy_score
import json
from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
```

```
def train(model,device,tokenizer,args):
    """
    训练模型
    Args:
        model: 模型
        device: 设备信息
        tokenizer: 分词器
        args: 训练参数配置信息

    Returns:
    """
    
    tb_write = SummaryWriter()
    #通过SentimentDataSet类构件训练所需的data_set
    train_data = SentimentDataSet(tokenizer, args.max_len, args.data_dir, "train_sentiment_analysis",
                                          args.train_file_path)
    
    #通过BatchSampler和DataLoader构件训练所需的迭代器
    train_sampler = BatchSampler(train_data, batch_size=args.train_batch_size, shuffle=True, drop_last=False)
    train_data_loader = DataLoader(train_data,batch_sampler=batch_sampler,collate_fn=collate_func_sentiment_analysis)
    #通过SentimentDataSet类构建测试所需的data_set
    test_data = SentimentDataSet(tokenizer, args.max_len, args.data_dir, "test_sentiment_analysis",
                                         args.test_file_path)
    
    #计算模型训练所需的总步数
    total_steps = len(train_data_loader) * args.num_train_epochs
    logger.info("总训练步数为:{}".format(total_steps))
    #将模型映射到指定的设备商
    model.to(device)
    
    #设置优化器
    scheduler = LinearDecayWithWarmup(args.learning_rate, total_steps, args.warmup_proportion)
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        epsilon=args.adam_epsilon,
        apply_decay_param_fun=lambda x: x in decay_params)
    
    model.train()
    tr_loss, logging_loss, max_acc = 0.0, 0.0, 0.0
    global_step = 0
    # 开始训练模型
    for iepoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            label = batch["label"]
            # 获取训练结果
            outputs = model.forward(input_ids, attention_mask, label)
            loss = outputs[0]
            tr_loss += loss.item()
            # 将损失值放到Iter中，方便观察
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            # 损失进行回传
            loss.backward()
            # 参数进行优化
            optimizer.step()
            scheduler.step()
            # 清空梯度
            optimizer.clear_grad()
            global_step += 1
            # 如果步数整除logging_steps，则记录学习率和训练集损失值
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_write.add_scalar("lr", scheduler.get_lr(), global_step)
                tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss
            # 如果步数整除save_model_steps，则进行模型测试，记录测试集的损失、准确率以及F1
            if args.save_model_steps > 0 and global_step % args.save_model_steps == 0:
                eval_loss, eval_acc, eval_f1 = evaluate(model, test_data, args)
                logger.info("eval_loss is {}, eval_acc is {} , eval_f1 is {}".format(eval_loss, eval_acc, eval_f1))
                tb_write.add_scalar("eval_loss", eval_loss, global_step)
                tb_write.add_scalar("eval_acc", eval_acc, global_step)
                tb_write.add_scalar("eval_f1", eval_f1, global_step)
                # 当eval_f1大于max_acc时，更新保存模型，并进行记录
                if eval_f1 >= max_acc:
                    max_acc = eval_f1
                    output_dir = os.path.join(args.output_dir, "checkpoint")
                    # 更新保存模型
                    model.save_pretrained(output_dir)
                    json_output_dir = os.path.join(output_dir, "json_data.json")
                    # 记录对应指标
                    fin = open(json_output_dir, "w", encoding="utf-8")
                    fin.write(json.dumps(
                        {"eval_loss": eval_loss, "eval_acc": eval_acc, "eval_f1": eval_f1, "global_step": global_step},
                        ensure_ascii=False, indent=4) + "\n")
                    fin.close()
                model.train()
```

### 设置模型训练参数，可根据自己的需要进行修改

#### 参数

训练参数可自行添加，包含参数具体如下：
| 参数                  | 类型  | 默认值                       | 描述                                               |
| --------------------- | ----- | ---------------------------- | -------------------------------------------------- |
| device                | str   | "0"                          | 设置设备编号                                       |
| train_file_path       | str   | "work/data/train.json"       | 训练集文件路径                                     |
| test_file_path        | str   | "work/data/test.json"        | 测试集文件路径                                     |
| pretrained_model_path | str   | "work/bert-paddle"           | 预训练模型路径                                     |
| vocab_path            | str   | "work/bert-paddle/vocab.txt" | 模型字典文件路径                                   |
| data_dir              | str   | "data"                       | 缓存文件保存路径                                   |
| num_train_epochs      | int   | 5                            | 训练轮数                                           |
| train_batch_size      | int   | 64                           | 训练的batch_size大小                               |
| test_batch_size       | int   | 32                           | 测试的batch_size大小                               |
| learning_rate         | float | 5e-5                         | 学习率                                             |
| warmup_proportion     | float | 0.1                          | warm up概率，即训练总步长的百分之多少，进行warm up |
| weight_decay          | float | 0.01                         | AdamW优化器的权重衰减系数                          |
| adam_epsilon          | float | 1e-8                         | AdamW优化器的epsilon值                             |
| logging_steps         | int   | 5                            | log记录步数                                        |
| save_model_steps      | int   | 200                          | 模型验证步数                                       |
| output_dir            | str   | "work/output_dir/"           | 模型输出路径                                       |
| seed                  | int   | 2020                         | 随机种子                                           |
| max_len               | int   | 256                          | 模型输入最大长度                                   |
| num_labels            | int   | 6                            | 标签个数                                           |

```
def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设备编号')
    parser.add_argument('--train_file_path', default='./data/train.json', type=str, help='训练集文件路径')
    parser.add_argument('--test_file_path', default='./data/test.json', type=str, help='测试集文件路径')
    parser.add_argument('--vocab_path', default="./bert-paddle/vocab.txt", type=str, help='模型字典文件路径')
    parser.add_argument('--pretrained_model_path', default="./bert-paddle", type=str, help='预训练模型路径')
    parser.add_argument('--data_dir', default='data/', type=str, help='缓存文件保存路径')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练轮数')
    parser.add_argument('--train_batch_size', default=2, type=int, help='训练的batch_size大小')
    parser.add_argument('--test_batch_size', default=2, type=int, help='测试的batch_size大小')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='学习率')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument("--weight_decay", default=0.01, type=float, help='AdamW优化器的权重衰减系数')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='AdamW优化器的epsilon值')
    parser.add_argument('--save_model_steps', default=200, type=int, help='模型验证步数')
    parser.add_argument('--logging_steps', default=5, type=int, help='log记录步数')
    parser.add_argument('--output_dir', default='./output_dir', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=2020, help='随机种子')
    parser.add_argument('--max_len', type=int, default=256, help='模型输入最大长度')
    parser.add_argument('--num_labels', type=int, default=6, help='标签个数')
    return parser.parse_args(args=[])
```

```
args = set_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# 获取device信息，用于模型训练
device = "gpu:{}".format(args.device) if paddle.fluid.is_compiled_with_cuda() and int(args.device) >= 0 else "cpu"
paddle.device.set_device(device)
# 设置随机种子，方便模型复现
if args.seed:
    paddle.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    
#加载预训练模型，进行模型初始化
model = Model.from_pretrained(args.pretrained_model_path, number_label=args.num_labels)
# 实例化tokenizer
tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
# 创建模型的输出目录
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
# 开始训练
train(model, device, tokenizer, args)
```

```
Iter (loss=0.011): 100%|█████████▉| 433/434 [03:30<00:00,  2.37it/s]
Iter (loss=0.052): 100%|█████████▉| 433/434 [03:30<00:00,  2.37it/s]
Iter (loss=0.052): 100%|██████████| 434/434 [03:30<00:00,  2.06it/s]
Epoch: 100%|██████████| 5/5 [17:50<00:00, 214.00s/it]
```

**训练完以后，有以下截图就代表训练完整**

![image-20211024112212826](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20211024112212826.png)

#### 总结

1. 熟悉paddlenlp
2. 熟悉paddle调用bert
3. 情感分类

## 预测

预测部分可参考上篇文章。

## 下载

所有代码以及模型已经上传百度网盘。

链接：https://pan.baidu.com/s/1uRYRmOPvyGhCQj0xuw8L1A 
提取码：6pfr

