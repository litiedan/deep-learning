from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


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

def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3),  activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model

model = baseline_model()

model.compile(
    optimizer='rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy']
)
model.fit(train_images,train_labels,epochs = 1,batch_size = 64)
test_loss,test_acc  = model.evaluate(test_images,test_labels)
print(test_acc)
print("Saving model to disk \n")
mp = "/home/lzq/mnist/model.h5"
model.save(mp)
