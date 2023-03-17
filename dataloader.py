import os
import time
from datetime import timedelta
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import pickle as pkl

MAX_VOCAB_SIZE = 10000  # length of vocabulary list limit
UNK, PAD = '<UNK>', '<PAD>'  # sign of UNKNOWN and PADDING


def preprocess(dataset_file_path):
    """
    This method is to precess the csv file and then get the contents and labels
    :param dataset_file_path: the file pat of dataset
    :return: contents, categoties of each content, dictionary that mapping the categories
    """
    dataframe = pd.read_csv(dataset_file_path)
    category = dataframe['category'].unique()

    y_label = {}
    with open('./dataset/class.txt', 'w', encoding='utf-8') as fp:
        for i in range(len(category)):
            y_label[category[i]] = i
            fp.write(category[i] + '\n')

    dataframe['category'] = dataframe['category'].map(y_label)
    X_data = dataframe['content'].to_numpy()
    Y_data = dataframe['category'].to_numpy()

    return X_data, Y_data, y_label


def build_vocab(X_data, tokenizer, max_size, min_freq):
    """
    This method is to build vocabulary list
    :param X_data: contents that generating vocabulary list
    :param tokenizer: a lambda function dividing the sentence and then get tokens
    :param max_size: the max size of vocabulary list
    :param min_freq: limit the minimum frequency of each vocabulary that appears
    :return: a vocabulary dictionary containing words, UNK and PAD
    """
    vocab_dic = {}

    for data in X_data:
        for word in tokenizer(data):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1

    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]

    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

    return vocab_dic


def build_dataset(config, use_word=True):
    """
    This method is to build the dataset
    :param config: config containing batch_size, vocab_path
    :param use_word:
    :return:
    """
    X_data, Y_data, y_label = preprocess(config.dataset_filepath)

    if use_word:
        tokenizer = lambda x: x.split(' ')  # word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(X_data, Y_data, pad_size=32):
        """
        This method is to load the dataset
        :param X_data: content data
        :param Y_data: label data
        :param pad_size: size of padding ensuring the length of each sentence are the same.
        :return: a tuple containing the
        """
        contents = []
        for idx in range(len(X_data)):
            content, label = X_data[idx], Y_data[idx]
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)

            if pad_size:
                # padding if the length of a sentence is less than the padding size
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
                # word to id
                for word in token:
                    # if the word is out of corpus identifying it as UNK
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    # Divide the dataset data into three parts: Train, Validate and Test
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_train, y_train, test_size=0.3)

    # Then get the processed dataset
    train = load_dataset(x_train, y_train, config.pad_size)
    validate = load_dataset(x_val, y_val, config.pad_size)
    test = load_dataset(x_test, y_test, config.pad_size)

    return vocab, train, validate, test


class DatasetIterater(object):
    """
    This class is used to iterate the dataset
    """
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # recording if the number of batch is Integer

        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # the length of a sentence before padding
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    """
    Build the iterator
    :param dataset: dataset
    :param config: configuration
    :return: an iterator instance
    """

    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """
    Get the running time
    :param start_time: the start time
    :return: The running time
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':

    dataset_filepath = './dataset/5g.csv'
    X_data, Y_data, y_label = preprocess(dataset_filepath)

    '''extracting pre-trained word vector'''
    vocab_dir = "./dataset/vocab.pkl"
    emb_dim = 300

    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        tokenizer = lambda x: x.split(' ')  # constructing the vocabulary list based on word size
        word_to_id = build_vocab(X_data, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))
