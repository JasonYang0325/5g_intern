# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):

    """config"""
    def __init__(self, embedding):
        self.model_name = 'TextRNN'
        self.dataset_filepath = 'dataset/5g.csv'
        self.class_list = [x.strip() for x in open('dataset/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = 'dataset/vocab.pkl'                                # 词表
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = 'dataset/data_logs/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('dataset/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # Random inactivation
        self.require_improvement = 1000                                 # If there has been no improvement after 1000
                                                                        # batches, end the training early.
        self.num_classes = len(self.class_list)                         # number of class
        self.n_vocab = 0                                                # number of vocabulary list
        self.num_epochs = 10                                            # number of epoch
        self.batch_size = 1                                             # mini-batch size
        self.pad_size = 32                                              # padding size
        self.learning_rate = 1e-3                                       # learning rate
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # The dimension of character embeddings should be consistent
                                                                        # if pre-trained word embeddings are used
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    """
    bi-LSTM + FC
    """
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out