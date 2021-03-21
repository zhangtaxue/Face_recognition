import numpy as np
from tensorflow.keras import layers
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def dataset(image_path, label_path):
    #首先读取数据集
    picture_in = open(image_path,"rb")
    picture = pickle.load(picture_in)
    label_in = open(label_path,"rb")
    label = pickle.load(label_in)

    # 输出训练集、验证集、测试集的数量
    picture = picture/255

    return picture, label

def train_model(picture, label):
    model = tf.keras.models.Sequential()
    print(picture.shape[1:])
    model.add(Conv2D(32, (3, 3), input_shape=picture.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) 

    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # 输出模型概况
    model.summary()

    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc'])

    history = model.fit(picture,label,batch_size=32 ,epochs = 5 ,validation_split= 0.2)
    model.save('new_my_mode.h5')
    return history

def draw_picture(history):
    loss = history.history['loss']  # 测试集损失
    acc = history.history['acc']  # 测试集准确率
    val_loss = history.history['val_loss']  # 验证集损失
    val_acc = history.history['val_acc']  # 验证集准确率
    plt.figure()
    plt.subplot(221)
    plt.plot(loss)
    plt.title('loss')
    plt.subplot(222)
    plt.plot(acc)
    plt.title('acc')
    plt.subplot(223)
    plt.plot(val_loss)
    plt.title('val_loss')
    plt.subplot(224)
    plt.plot(val_acc)
    plt.title('val_acc')
    plt.show()

if __name__ == '__main__':
    image_path = "D:\\zyc_dm\\Face_recognition\\new_image.pickle"
    label_path = "D:\\zyc_dm\\Face_recognition\\new_image_label.pickle"
    picture, label = dataset(image_path, label_path)
    history = train_model(picture, label)
    draw_picture(history)
