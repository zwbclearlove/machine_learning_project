"""
模型建立，训练
"""
import numpy as np
import h5py
from skimage import io
import matplotlib.pyplot as plt
from keras.metrics import CategoricalAccuracy
from keras.models import Sequential
from keras.optimizers import Adam

from keras.layers import Conv2D, MaxPool2D, AvgPool2D
from keras.layers import Dense, Flatten, Dropout, BatchNormalization

from keras.utils import to_categorical
from keras.datasets import mnist
import keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()

data = h5py.File('all_data.h5', 'r', driver='core')

img_x, img_y = 48, 48

all_data = data['data_pixel']
all_val = data['data_label']
test_data = data['test_data']
tmp = np.ndarray(shape=all_data.shape)
for i, v in enumerate(all_data):
    tmp[i] = v
all_data = tmp
tmp = np.ndarray(shape=test_data.shape)
for i, v in enumerate(test_data):
    tmp[i] = v
test_data = tmp


all_val = to_categorical(all_val, 7)

print(all_val[0])

all_data = all_data.reshape(all_data.shape[0], img_x, img_y, 1)
all_data = all_data.astype('float32')
all_data /= 255
test_data = test_data.reshape(test_data.shape[0], 48, 48, 1)
test_data = test_data.astype('float32')
test_data /= 255
print(all_data.shape)
print(all_val.shape)
'''
第一次尝试使用的模型
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(img_x, img_y, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[CategoricalAccuracy()])
model.summary()
'''
'''
第二次尝试
model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(MaxPool2D(pool_size=(5, 5), strides=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))
'''
'''
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))
'''

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              loss='categorical_crossentropy',
              metrics=[CategoricalAccuracy()])
model.summary()

model.fit(all_data, all_val, validation_data=(all_data, all_val), batch_size=64, epochs=20)

score, acc = model.evaluate(all_data, all_val)
print('\nAccuracy: ', acc * 100)

model.save('my_model_vgg1.h5')






