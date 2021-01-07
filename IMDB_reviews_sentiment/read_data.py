import numpy as np
import pandas as pd
import re
import string



def clean_text(text):
    punctuation = "\"-+*/.,?!:;(){}[]"
    text = text.lower().replace('\n', '')
    text = text.replace('<br />', ' ')
    text = text.replace('‘', '\'')

    for c in punctuation:
        text = text.replace(c, ' ')
    return text


def labelize_reviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append((v, [label]))
    return labelized


if __name__ == '__main__':
    '''
    dr = re.compile(r'<[^>]+>', re.S)
    df = pd.read_csv('train.csv')

    d = df.values[0:20]
    dd = pd.DataFrame(d, columns=["id", "review", "sentiment"])
    x_train = []
    for line in dd.values:
        review = line[1]
        # print(review)
        new_review = review
        # print(new_review)
        x_train.append(new_review)

    x_train = labelize_reviews(x_train, "TRAIN")
    print(x_train)
    # dd.to_csv("./new.csv", index=False)
    '''

    d = pd.read_csv('test_data.csv')
    labels_dict = {'negative': 0, 'positive': 1}
    d['sentiment'] = d['sentiment'].apply(lambda x: labels_dict[x])
    d['review'] = d['review'].apply(lambda x: clean_text(x))
    d.to_csv("./test_data_new.csv", index=False)

