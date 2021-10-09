import os  
import random  
  
val_percent = 0.1 #验证集(实际为 0.9*0.1)
train_percent = 0.9 #训练集 测试集为1-0.9
xmlfilepath = 'Annotations'  
txtsavepath = 'ImageSets\Main'  
total_xml = os.listdir(xmlfilepath)  
  
num=len(total_xml)  
list=range(num)  
tr=int(num*train_percent)  
tv=int(tr*val_percent)  
train= random.sample(list,tr)  
val=random.sample(train,tv)  
  
ftest = open('ImageSets/Main/test.txt', 'w')  
ftrain = open('ImageSets/Main/train.txt', 'w')  
fval = open('ImageSets/Main/val.txt', 'w')  
  
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in train:  
        ftrain.write(name) 
        if i in val:   
            fval.write(name) 
    else:   
        ftest.write(name)  
  
ftrain.close()  
fval.close()  
ftest.close()