import h5py
import numpy as np
import keras
from keras.models import Model, load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pydot
import graphviz
(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

# 模型地址
MODEL_PATH = '/home/lzq/mnist/model.h5'
model = load_model(MODEL_PATH, compile=False)

print('the number of layers in this model:'+str(len(model.layers)))
print('\nModel loaded ({0}).'.format(MODEL_PATH))
input = test_images[0:9]
input = input.reshape((9,28,28,1))
# print(input.shape)
predictions = model.predict(input)
print(predictions.shape) 
print(predictions) 

# 这个subplot函数的作用是确定图像的位置以及图像的个数，前两个3的意思是可以放9张图片，如果变成221的话，就是可以放4张图片，然后后面的1，是确定图像的位置，处于第一个，以下的subplot同理
# 这里个把图片显示出来
# X_train存储的是图像的像素点组成的list数据类型，这里面又由一个二维的list（28 x 28的像素点值）和一个对应的标签list组成，y_train存储的是对应图像的标签，也就是该图像代表什么数
plt.subplot(331)
plt.imshow(test_images[0], cmap=plt.get_cmap('gray'))
plt.subplot(332)
plt.imshow(test_images[1], cmap=plt.get_cmap('gray'))
plt.subplot(333)
plt.imshow(test_images[2], cmap=plt.get_cmap('gray'))
plt.subplot(334)
plt.imshow(test_images[3], cmap=plt.get_cmap('gray'))
plt.subplot(335)
plt.imshow(test_images[4], cmap=plt.get_cmap('gray'))
plt.subplot(336)
plt.imshow(test_images[5], cmap=plt.get_cmap('gray'))
plt.subplot(337)
plt.imshow(test_images[6], cmap=plt.get_cmap('gray'))
plt.subplot(338)
plt.imshow(test_images[7], cmap=plt.get_cmap('gray'))
plt.subplot(339)
plt.imshow(test_images[8], cmap=plt.get_cmap('gray'))
plt.show()