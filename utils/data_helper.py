import numpy as np
import re
from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn
import pdb
import collections

# 过滤函数
def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# 加载数据的函数
def load_data_and_labels(positive_data_file, negative_data_file):

    # 加载数据（使用 urf-8 存在问题）
    positive = open(positive_data_file, 'rb').read().decode('ISO-8859-1')
    negative = open(negative_data_file, 'rb').read().decode('ISO-8859-1')

    # 去掉换行符
    positive_examples = positive.split('\n')[:-1]
    negative_examples = negative.split('\n')[:-1]

    # 去掉开始和结尾的空格
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]

    # 组合样本
    x_text = positive_examples + negative_examples
    # 过滤样本
    x_text = [clean_str(sent) for sent in x_text]

    # 生成 label
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

# 加载 labels
def load_data_labels(data_file, labels_file):
    data = []
    labels = []
    with open(data_file, 'r', encoding='latin-1') as f:
        data.extend([s.strip()] for s in f.readlines())
        data = [clean_str(s) for s in data]

    with open(labels_file, 'r') as f:
        labels.extend([s.strip()] for s in f.readlines())
        labels = [labels.split(',')[1].strip() for lable in labels]

    # label 二值化
    label = LabelBinarizer()
    y = label.fit_transform(labels)

    # 生成 1000 维的映射字典
    vocab_processor = learn.preprocessing.VocabularyProcessor(1000)
    x = np.array(list(vocab_processor.fit_transform(data)))
    return x, y, vocab_processor

# 生成 batch 数据
def batch_iter(data, batch_size, num_epochs, shuffle = True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # 每个 epoch 进行 shuffle
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]

