PIL和cv2是python中两个常用的图像处理库，PIL一般是anaconda自带的，cv2是opencv的python版本。base64在网络传输图片的时候经常用到。

##PIL读取、保存图片方法
from PIL import Image
img = Image.open(img_path)
img.save(img_path2)


##cv2读取、保存图片方法
import cv2
img = cv2.imread(img_path)
cv2.imwrite(img_path2, img)


##图片文件打开为base64
import base64

def img_base64(img_path):
with open(img_path,"rb") as f:
    base64_str = base64.b64encode(f.read())
return base64_str 
1、PIL和cv2转换

##PIL转cv2
import cv2
from PIL import Image
import numpy as np

def pil_cv2(img_path):
image = Image.open(img_path)
img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
return img


##cv2转PIL
import cv2
from PIL import Image

def cv2_pil(img_path):
image = cv2.imread(img_path)
image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
return image
2、PIL和base64转换

##PIL转base64
import base64
from io import BytesIO

def pil_base64(image):
img_buffer = BytesIO()
image.save(img_buffer, format='JPEG')
byte_data = img_buffer.getvalue()
base64_str = base64.b64encode(byte_data)
return base64_str


##base64转PIL
import base64
from io import BytesIO
from PIL import Image

def base64_pil(base64_str):
image = base64.b64decode(base64_str)
image = BytesIO(image)
image = Image.open(image)
return image
3、cv2和base64转换

复制代码
##cv2转base64
import cv2

def cv2_base64(image):
base64_str = cv2.imencode('.jpg',image)[1].tostring()
base64_str = base64.b64encode(base64_str)
return base64_str 


##base64转cv2

import base64
import numpy as np
import cv2

def base64_cv2(base64_str):
imgString = base64.b64decode(base64_str)
nparr = np.fromstring(imgString,np.uint8) 
image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
return image