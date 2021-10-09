'''
opencv每隔若干秒拍照并且保存
'''
import os
import time
import threading
from cv2 import cv2 as cv2 #在vscode里面直接import cv2会显示代码有问题，其实是没问题的，这种导入方式可以避免；

def takephoto():
    cap = cv2.VideoCapture(0)
    index = 0
    ret, frame = cap.read()
    while ret:
        for index in range(100):
            resize = cv2.resize(frame, (512,512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(index)+'.jpg', resize)
            time.sleep(0.5)
            ret, frame = cap.read()
            index += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0 

if __name__=='__main__':
    print('Begin to take pictures..........')
    takephoto()
    print('Finished !!')
