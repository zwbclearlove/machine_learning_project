"""
    将评论转化为向量
"""
import pandas as pd
import numpy as np
import re
import string
import h5py

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


word_dic = {}
for review in data['review']:
    wl = tokenize(review)
    print(wl)
    for word in wl:
        if word_dic.get(word):
            word_dic[word] += 1
        else:
            word_dic[word] = 1
    break

for review in test['review']:
    wl = tokenize(review)
    print(wl)
    for word in wl:
        if word in word_dic:
            word_dic[word] += 1
        else:
            word_dic[word] = 1
    break

word_dic = sorted(word_dic.items(), key=lambda x: x[1], reverse=True)
print(word_dic)
word_dic = [word[0] for word in word_dic]
print(word_dic)


def get_vec(review):
    vec = []
    num = 0
    word_list = tokenize(review)
    for word in word_list:
        if word in word_dic:
            vec.append(word_dic.index(word))
        else:
            vec.append(0)
        num += 1
        if num == 300:
            break
    if num < 300:
        for i in range(300- num):
            vec.append(0)
    return vec


x_train = []
y_train = []
test = []

for review in data['review']:
    x_train.append(get_vec(review))

for label in data['sentiment']:
    if label == 'positive':
        y_train.append(1)
    else:
        y_train.append(0)

for review in test['review']:
    test.append(get_vec(review))

data_set = h5py.File('data.h5', 'w', driver='core')
data_set.create_dataset('train_reviews', dtype='int32', data=x_train)
data_set.create_dataset('train_label', dtype='uint8', data=x_train)
data_set.create_dataset('test_reviews', dtype='int32', data=test)
data_set.close()

print('save data finish')
