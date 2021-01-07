"""
利用生成的模型进行预测
"""

import csv
import os
import numpy as np
import h5py
import pandas as pd
from keras.models import load_model

data = h5py.File('all_data.h5', 'r', driver='core')
test_data = data['test_data']

tmp = np.ndarray(shape=test_data.shape)
for i, v in enumerate(test_data):
    tmp[i] = v
test_data = tmp


test_data = test_data.reshape(test_data.shape[0], 48, 48, 1)
test_data = test_data.astype('float32')
test_data /= 255
print(test_data.shape)

model = load_model('my_model_vgg1.h5')

predict = model.predict(test_data)
ans = []
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
for p in predict:
    index = np.argmax(p)
    ans.append(class_names[index])


sub = pd.read_csv('data/submission.csv')

sub['class'] = ans
sub.to_csv('submission.csv', index=False)




