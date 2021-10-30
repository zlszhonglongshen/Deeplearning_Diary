# Labelme工具标注格式转化为PaddleSeg格式

### labelme标注

关于该软件的详细介绍可以参考以下两篇文章，这里不做详细说明。

* https://blog.csdn.net/u014061630/article/details/88756644
* https://zhuanlan.zhihu.com/p/112512069

### json格式转数据文件格式

**转换后的格式如下**

```
img.png 
label.png 
label_names.txt 
label_viz.png
```

#### 转换代码

```python
import os，glob
json_file = glob.glob('Dataset/*.json')#匹配文件，返回一个列表
for file in json_file:
    os.system("labelme_json_to_dataset %s"%(file))#命令行json转数据集文件
```

转换成功截图，每个文件夹下包含的文件就是上面的4个文件。

![image-20211029130046961](/Users/zhongls/Library/Application Support/typora-user-images/image-20211029130046961.png)

### labelme格式转PaddleSeg格式

查看训练样本数量

```python
images_path = "/dataset" #设置数据集路径
image_count = len([os.path.join(images_path, image_name) 
          for image_name in os.listdir(images_path)])
print("用于训练的图片样本数量:", image_count)
```

转换函数

```python
def _sort_images(image_dir):
    """
    对文件夹内的图像进行按照文件名排序
    """
    images = []
    labels = []

    for image_name in os.listdir(image_dir):
        if os.path.isdir(os.path.join(image_dir, image_name)):
            images.append(os.path.join(os.path.join(image_dir, image_name), 'img.png'))
            labels.append(os.path.join(os.path.join(image_dir, image_name), 'label.png'))

    return sorted(images), sorted(labels)
"""
这里的分割符是\t，后面使用PaddleSeg的时候要注意修改相关代码，因为PaddleSeg读取文件时默认的分割符是空格。
当然也可以将这里的\t替换为空格。
"""
def write_file(mode, images, labels):
    with open('./{}.txt'.format(mode), 'w') as f:
        for i in range(len(images)):
            #f.write('{}\t{}\n'.format(images[i], labels[i]))
            f.write('{} {}\n'.format(images[i], labels[i]))            
    
"""
由于所有文件都是散落在文件夹中，在训练时我们需要使用的是数据集和标签对应的数据关系，
所以我们第一步是对原始的数据集进行整理，得到数据集和标签两个数组，分别一一对应。
这样可以在使用的时候能够很方便的找到原始数据和标签的对应关系，否则对于原有的文件夹图片数据无法直接应用。
"""
images, labels = _sort_images(images_path)
eval_num = int(image_count * 0.15)

"""
由于图片数量有限，这里的测试集和验证集采用相同的一组图片。
"""
write_file('train', images[:-eval_num], labels[:-eval_num])
write_file('test', images[-eval_num:], labels[-eval_num:])
write_file('eval', images[-eval_num:], labels[-eval_num:])
```

#### 看下格式对不对

```python
with open('./train.txt', 'r') as f:
    i = 0

    for line in f.readlines():
        image_path, label_path = line.strip().split(' ')
        image = np.array(PilImage.open(image_path))
        label = np.array(PilImage.open(label_path))
    
        if i > 2:
            break
        # 进行图片的展示
        plt.figure()

        plt.subplot(1,2,1), 
        plt.title('Train Image')
        plt.imshow(image.astype('uint8'))
        plt.axis('off')

        plt.subplot(1,2,2), 
        plt.title('Label')
        plt.imshow(label.astype('uint8'), cmap='gray')
        plt.axis('off')

        plt.show()
        i = i + 1
```

![image-20211029130936434](/Users/zhongls/Library/Application Support/typora-user-images/image-20211029130936434.png)



### 总结

关于数据转换就到这里就结束，**下期利用paddleseg实现自己的图像分割项目**



### 参考

* https://blog.csdn.net/weixin_45693265/article/details/116320107?spm=1001.2014.3001.5501

