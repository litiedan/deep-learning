from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models

from keras.models import Model
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Flatten,Dense
from keras.layers import Input

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

print("train_images.shape:",train_images.shape)
print("train_labels.shape",train_labels.shape)
print("test_images.shape:",test_images.shape)
print("test_labels.shape",test_labels.shape)

train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


x = Input(shape = (28,28,1))
conv1 = Conv2D(32, (3, 3),activation='relu')(x)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3),activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(64, (3, 3),activation='relu')(pool2)
flat = Flatten()(conv3)
den1 = Dense(64, activation='relu')(flat)
output =  Dense(10, activation='softmax')(den1)
model = Model(inputs = x,outputs = output)
model.summary()


model.compile(
    optimizer='rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy']
)
model.fit(train_images,train_labels,epochs = 1,batch_size = 64)
test_loss,test_acc  = model.evaluate(test_images,test_labels)
print(test_acc)
print("Saving model to disk \n")
mp = "/home/lzq/mnist/model.h5"
model.save(mp)