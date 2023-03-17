# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """config"""
    def __init__(self, embedding):
        self.model_name = 'TextCNN'
        self.dataset_filepath = 'dataset/5g.csv'
        self.class_list = [x.strip() for x in open('dataset/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = 'dataset/vocab.pkl'  # 词表
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = 'dataset/data_logs/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('dataset/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5                                              # Random inactivation
        self.require_improvement = 1000                                 # If there has been no improvement after 1000 batches, end the training early.
        self.num_classes = len(self.class_list)                         # number of class
        self.n_vocab = 0                                                # number of vocabulary list
        self.num_epochs = 20                                            # number of epoch
        self.batch_size = 1                                             # mini-batch size
        self.pad_size = 32                                              # padding size
        self.learning_rate = 1e-3                                       # learning rate
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # The dimension of character embeddings should be consistent if pre-trained word embeddings are used
        self.filter_sizes = (2, 3, 4)                                   # The size of conv filter
        self.num_filters = 256                                          # The number of conv filter


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out