"""
调用keras深度学习库进行实现
"""
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Flatten, Conv1D, Dropout, Activation
from keras.preprocessing.text import Tokenizer


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import re
import string

# For reproducibility
from numpy.random import seed

# CPU
BATCH_SIZE = 32
EPOCHS = 5
VOCAB_SIZE = 20000
MAX_LEN = 300
EMBEDDING_DIM = 40

LABELS = ['negative', 'positive']

data = pd.read_csv('train_new.csv')
test = pd.read_csv('test_data_new.csv')
all_data = data.append(test)
print(data.dtypes)
print("Train shape (rows, columns): ", data.shape)
print(test.dtypes)
print("Test shape (rows, columns): ", test.shape)

print("\n--- First Sample ---")
print('Label:', data['sentiment'][0])
print('Text:', data['review'][0])

# Custom Tokenizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


"""
plt.hist([len(tokenize(s)) for s in data['review'].values], bins=50)
plt.title('Tokens per sentence')
plt.xlabel('Len (number of token)')
plt.ylabel('# samples')
plt.show()
"""
# 数据预处理
imdb_tokenizer = Tokenizer(num_words=VOCAB_SIZE)
imdb_tokenizer.fit_on_texts(all_data['review'].values)

x_train_seq = imdb_tokenizer.texts_to_sequences(data['review'].values)
x_val_seq = imdb_tokenizer.texts_to_sequences(data['review'].values)

x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_LEN, padding="post", value=0)
x_val = sequence.pad_sequences(x_val_seq, maxlen=MAX_LEN, padding="post", value=0)

test_seq = imdb_tokenizer.texts_to_sequences(test['review'].values)
test_set = sequence.pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", value=0)

y_train, y_val = data['sentiment'].values, data['sentiment'].values

for i in range(1):
    print(str(i) + ' sample before preprocessing: \n', data['review'].values[i], '\n')
    print(str(i) + ' sample after preprocessing: \n', x_train[i])

# Model Parameters - You can play with these

NUM_FILTERS = 250
KERNEL_SIZE = 3
HIDDEN_DIMS = 250

# CNN Model
print('Build model...')
'''
第一次尝试模型
model = Sequential()
model.add(Embedding(20000, 32, input_length=MAX_LEN))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
'''
# 第二次尝试模型
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
model.add(Dropout(0.2))
model.add(Conv1D(NUM_FILTERS, KERNEL_SIZE, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(HIDDEN_DIMS))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, validation_split=0.2,batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)

# Evaluate the model
score, acc = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE)
print('\nAccuracy: ', acc * 100)

model.save('my_model.h5')

predict = model.predict(test_set)
ans = []
for i in predict:
    if i > 0.5:
        ans.append('positive')
    else:
        ans.append('negative')

sub = pd.read_csv('submission.csv')

sub['sentiment'] = ans
sub.to_csv('subb_new.csv', index=False)
