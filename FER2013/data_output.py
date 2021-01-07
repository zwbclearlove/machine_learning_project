import csv
import os
import numpy as np
import h5py
from skimage import io
import pandas as pd

data = h5py.File('all_data.h5', 'r', driver='core')

all_data = data['data_pixel']
all_val = data['data_label']

print('image : ', all_data[0])
print('label : ', all_val[0])
