# coding: UTF-8
import time
import torch
import numpy as np
import argparse
from train_eval import train, init_network
from importlib import import_module
from dataloader import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='5g text classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextRNN')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':

    # Sougou News: embedding_SougouNews.npz, Tencent: embedding_Tencent.npz, Random: random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model  # 'TextRNN'  # TextCNN, TextRNN

    x = import_module('models.' + model_name)
    config = x.Config(embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # ensuring that every result is the same

    start_time = time.time()
    print("Loading data...")

    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)

    # Get the iterator of training set, validation set and test set
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)

    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)

    train(config, model, train_iter, dev_iter, test_iter)
