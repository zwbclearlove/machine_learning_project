"""
对数据进行预处理
"""

import csv
import os
import numpy as np
import h5py
from skimage import io
import pandas as pd

ck_path = 'data\\train'

angry_path = os.path.join(ck_path, 'angry')
disgust_path = os.path.join(ck_path, 'disgust')
fear_path = os.path.join(ck_path, 'fear')
happy_path = os.path.join(ck_path, 'happy')
neutral_path = os.path.join(ck_path, 'neutral')
sad_path = os.path.join(ck_path, 'sad')
surprise_path = os.path.join(ck_path, 'surprise')
test_data_path = 'data/test'

angry_num = 0
disgust_num = 0
fear_num = 0
happy_num = 0
neutral_num = 0
sad_num = 0
surprise_num = 0

data_x = []
data_y = []
test_data = []

files = os.listdir(angry_path)
for filename in files:
    image = io.imread(os.path.join(angry_path, filename))
    data_x.append(image)
    data_y.append(0)
    angry_num += 1

files = os.listdir(disgust_path)
for filename in files:
    image = io.imread(os.path.join(disgust_path, filename))
    data_x.append(image)
    data_y.append(1)
    disgust_num += 1

files = os.listdir(fear_path)
for filename in files:
    image = io.imread(os.path.join(fear_path, filename))
    data_x.append(image)
    data_y.append(2)
    fear_num += 1

files = os.listdir(happy_path)
for filename in files:
    image = io.imread(os.path.join(happy_path, filename))
    data_x.append(image)
    data_y.append(3)
    happy_num += 1

files = os.listdir(neutral_path)
for filename in files:
    image = io.imread(os.path.join(neutral_path, filename))
    data_x.append(image)
    data_y.append(4)
    neutral_num += 1

files = os.listdir(sad_path)
for filename in files:
    image = io.imread(os.path.join(sad_path, filename))
    data_x.append(image)
    data_y.append(5)
    sad_num += 1

files = os.listdir(surprise_path)
for filename in files:
    image = io.imread(os.path.join(surprise_path, filename))
    data_x.append(image)
    data_y.append(6)
    surprise_num += 1

files = pd.read_csv('data/submission.csv')['file_name']
i = 10
for filename in files:
    if i > 0:
        print(filename)
        i -= 1
    image = io.imread(os.path.join(test_data_path, filename))
    test_data.append(image)

print(len(data_x))
print(len(data_y))

print('angry :', angry_num)
print('disgust :', disgust_num)
print('fear :', fear_num)
print('happy :', happy_num)
print('neutral :', neutral_num)
print('sad :', sad_num)
print('surprise :', surprise_num)
print('test_data :', len(test_data))

data_file = h5py.File('all_data.h5', 'w')
data_file.create_dataset('data_pixel', dtype='uint8', data=data_x)
data_file.create_dataset('data_label', dtype='int64', data=data_y)
data_file.create_dataset('test_data', dtype='uint8', data=test_data)
data_file.close()



print('save data finish')
