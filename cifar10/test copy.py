from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import backend
# import os
import tensorflow as tf
from keras.models import Model
from keras.preprocessing import sequence
from keras.datasets import imdb
from matplotlib import pyplot as plt
 
from keras import backend as K
from keras.engine.topology import Layer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"#（其中0.1是选择所调用的gpu）
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
backend.set_image_data_format('channels_first')


# 设定随机数种子
seed = 7
np.random.seed(seed)


# 导入数据
(x_train, y_train), (x_validation, y_validation) = cifar10.load_data()
x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_train = x_train/255.0
print(x_train[0].shape)
x_validation = x_validation/255.0

# 进行one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]

class Self_Attention(Layer):
 
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
 
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它
 
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
 
        print("WQ.shape",WQ.shape)
 
        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
 
 
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
 
        QK = QK / (64**0.5)
 
        QK = K.softmax(QK)
 
        print("QK.shape",QK.shape)
 
        V = K.batch_dot(QK,WV)
 
        return V
 
    def compute_output_shape(self, input_shape):
 
        return (input_shape[0],input_shape[1],self.output_dim)
def hw_flatten(x):
    return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])
 
def pam(x):
    from keras.layers.convolutional import Conv2D
    from keras.constraints import maxnorm
    from keras.layers import BatchNormalization
    from keras.activations import relu
    def hw_flatten(x):
        return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])
    print('ssssssssssssssssssssss')
    print(K.shape(x))
    f = Conv2D(3,(1, 1), padding='same', kernel_constraint=maxnorm(3))(x)
    f = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(f)
    # f = relu(f, alpha=0.0, max_value=None, threshold=0.0)
    g = Conv2D(3,(1, 1), padding='same', kernel_constraint=maxnorm(3))(x)
    g = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(g)
    # g = relu(g, alpha=0.0, max_value=None, threshold=0.0)
    h = Conv2D(3,(3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
    h = Conv2D(3,(1, 1), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
    # gg = K.reshape(g, shape=[K.shape(g)[0], K.shape(g)[1]*K.shape(g)[2], K.shape(g)[3]])
    # ff = K.reshape(f, shape=[K.shape(f)[0], K.shape(f)[1]*K.shape(f)[2], K.shape(f)[3]])
    # hh = K.reshape(h, shape=[K.shape(h)[0], K.shape(h)[1]*K.shape(h)[2], K.shape(h)[3]])

    s = K.batch_dot(hw_flatten(g), K.permute_dimensions(hw_flatten(f), (0, 2, 1))) #[bs, N, N]
    # s = K.batch_dot(gg, K.permute_dimensions(ff, (0, 2, 1))) #[bs, N, N]

    beta = K.softmax(s, axis=-1)  # attention map
    o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]
    # o = K.batch_dot(beta, hh)  # [bs, N, C]

    o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
    # x =  gamma * o + x
    x = o + x
    return x
# # 定义模型创建函数
# def create_model(epochs=25):
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Self_Attention(name='scale',output_dim = 3))
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))
#     lrate = 0.01
#     decay = lrate/epochs
#     sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#     # 编译模型
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#     return model
# model = create_model(epochs)

from keras.layers import Input,Lambda
inputs = Input(shape=(3,32,32)) 
x = Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3))(inputs) 
x = Dropout(0.2)(x)
x = Conv2D(3, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Self_Attention(output_dim = 3)(x)
x = Lambda(pam)(x)
x = Flatten()(x) 
x = Dense(512, activation='relu', kernel_constraint=maxnorm(3))(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs, name="mnist_model")
epochs = 1
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# 训练模型及评估模型
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=32, verbose=2)
score = model.evaluate(x=x_validation, y=y_validation, verbose=0)
print('Accuracy: %.2f%%' % (score[1] * 100))
# save model
print("Saving model to disk \n")
mp = "/home/lzq/file/cifar10/iris_model.h5"
model.save(mp)

# # ###########################################测试##########################

import h5py
import numpy as np
from keras.models import Model, load_model
# 模型地址
MODEL_PATH = '/home/lzq/file/cifar10/iris_model.h5'
model = load_model(MODEL_PATH, compile=False)
print('the number of layers in this model:'+str(len(model.layers)))#621
print('\nModel loaded ({0}).'.format(MODEL_PATH))
input = x_train[0:9]
print(input.shape)
predictions = model.predict(input)
print(predictions.shape) 



# #剪裁网络模型
# model = Model(inputs=model.input, outputs=model.get_layer(index = -9).output)
# print('the number of layers in this model:'+str(len(model.layers)))

# print('Model loaded')

# # # Compute results
# predictions = model.predict(input)
# # #matplotlib problem on ubuntu terminal fix
# # #matplotlib.use('TkAgg')   
# print(predictions.shape)
# #
# #
# #(9, 32, 32, 32)  -8
# #(9, 32, 32, 32)  -7
# #(9, 32, 32, 32)  -6
# #(9, 32, 16, 16)  -5
# #(9, 8192)        -4
# #(9, 512)         -3
# #(9, 512)         -2
# #(9, 10)          -1


###############################
# #(9, 32, 32, 32)  -8
# #(9, 32, 32, 32)  -7
# #(9, 32, 32, 32)  -6
# #(9, 32, 16, 16)  -5
# #(9, 32, 8, 8)  -5
# #(9, 32, 4, 4)  -5
# #(9, 8192)        -4
# #(9, 512)         -3
# #(9, 512)         -2
# #(9, 10)          -1
# #(9, 32, 32, 32)  0
