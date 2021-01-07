"""
对未完全拟合得模型继续进行训练
"""
import numpy as np
import h5py
from skimage import io
import matplotlib.pyplot as plt
from keras.metrics import CategoricalAccuracy
from keras.models import Sequential, load_model

from keras.layers import Conv2D, MaxPool2D, AvgPool2D
from keras.layers import Dense, Flatten, Dropout

from keras.utils import to_categorical

data = h5py.File('all_data.h5', 'r', driver='core')

img_x, img_y = 48, 48

all_data = data['data_pixel']
all_val = data['data_label']
tmp = np.ndarray(shape=all_data.shape)
for i, v in enumerate(all_data):
    tmp[i] = v
all_data = tmp


all_val = to_categorical(all_val, 7)

print(all_val[0])

all_data = all_data.reshape(all_data.shape[0], img_x, img_y, 1)
all_data = all_data.astype('float32')
all_data /= 255
print(all_data.shape)
print(all_val.shape)

model = load_model('my_model_vgg1.h5')

model.fit(all_data, all_val, validation_data=(all_data, all_val), batch_size=64, epochs=10)

score, acc = model.evaluate(all_data, all_val)
print('\nAccuracy: ', acc * 100)

model.save('my_model_vgg1.h5')
