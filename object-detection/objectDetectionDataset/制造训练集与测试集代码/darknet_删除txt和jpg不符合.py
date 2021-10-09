#coding:utf-8
import os 

def get_fileNames(rootdir,type):
    fs = []
    for root,dirs,files in os.walk(rootdir,topdown=True):
        for name in files:
            _,ending = os.path.splitext(name)
            if ending==type:
                fs.append(name.split(".")[0])

    return fs


root_dir = "C:/Users/zhong/Desktop/hand"
jpglist = set(get_fileNames(root_dir,".jpg"))
txtlist = set(get_fileNames(root_dir,".txt"))


for i in list(jpglist^txtlist):
    i = str(i)+".jpg"
    os.remove(os.path.join(root_dir,i))