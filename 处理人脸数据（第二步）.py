import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import cv2
import pickle

size = 65
zyc = []
zyc_label = []
wx = []
wx_label = []

for i in range(2001):
    name1 = 'D:\\zyc_dm\\sy\\zyc_picture\\' + str(i) +'.jpg'
    zyc.append(name1)
    zyc_label.append(0)
    name2 = 'D:\\zyc_dm\\sy\\wx_picture\\' + str(i) + '.jpg'
    wx.append(name2)
    wx_label.append(1)
    print("There are %d zyc\nThere are %d wx" % (len(zyc), len(wx)))

#  打乱文件的顺序
#  先将图片地址和标签统一在一起
image_list = np.hstack((zyc,wx))
label_list = np.hstack((zyc_label,wx_label))
temp = np.array([image_list, label_list])  # 将位置和标签对应在一起，这样后面乱序的时候也是对应的
temp = temp.transpose()  # 转置
np.random.shuffle(temp)  # 乱序

# 打乱后重新将他们分开保存
image_list = list(temp[:, 0])
label_list = list(temp[:, 1])
label_list = [int(i) for i in label_list]  # 将字符型转化为整型
label_list = np.array(label_list)

#  将图片以相同尺寸大小保存
train_array = []
for img_path in tqdm(image_list):
    print(img_path)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 打开文件，我们的list里面储存的就是路径。
    # 我们将图片bian'成灰度以便减少计算量
    new_array = cv2.resize(img_array, (size, size))  # 大小转换
    train_array.append(new_array)

#  转化格式
picture = np.array(train_array).reshape(-1, size, size, 1)
label   = label_list

#  将图片保存为pickle文件，以便我们以后训练用
pickle_out = open("D:\\zyc_dm\\sy\\new_image.pickle","wb")
pickle.dump(picture, pickle_out)
pickle_out.close()
pickle_out = open("D:\\zyc_dm\\sy\\new_image_label.pickle","wb")
pickle.dump(label, pickle_out)
pickle_out.close()




