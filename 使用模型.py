import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
from PIL import Image,ImageOps,ImageDraw,ImageFont
import matplotlib.pyplot as plt


model = keras.models.load_model('new_my_mode.h5')     # 载入模型


def pd(image):
    im = Image.open(image)           # 读取图片路径
    im = im.resize((65, 65))         # 调整大小和模型输入大小一致
    im_grey = im.convert('L')
    tp = np.array(im_grey)           # 获取像素
    ret = model.predict((tp / 255).reshape((1, 65, 65, 1)))
    return ret


image_path ='zyc.jpg'
im=Image.open(image_path)
draw = ImageDraw.Draw(im)
textsize = 120
ft = ImageFont.truetype("Restu-Bundah-2.otf", textsize)
if pd(image_path) <= 0.5:
    draw.text((30, 30), u"zyc", font=ft)
else :
    draw.text((30, 30), u"wx", font=ft)
im.show()


