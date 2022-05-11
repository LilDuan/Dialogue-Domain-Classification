import math
import multiprocessing

import joblib
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import *

class MyDataset(Dataset):
    def __init__(self, params, path, vocab):

        self.data = load_from_csv(path)

        self.vocab = vocab
        self.device = params.device
        self.tokenizer = params.tokenizer
        self.min_freq = params.min_freq
        self.language = params.language

        self.word2idx = params.word2idx

    def __getitem__(self,index):

        content = self.data['text'][index]
        label = self.data['label'][index] - 1 #
        word2idx = self.word2idx

        # 构造词典时已经筛选过词频小于5的词汇

        words = sentence_process(content, tokenizer=self.tokenizer, language=self.language)

        words = [word2idx.get(w, word2idx['unk']) for w in words]   # 将word用数字表示

        return words, label

    def __len__(self):

        return len(self.data['text'])


def collate_fn(batch):
    """
    根据一个batch内的数据动态padding

    args:
        batch: [[input_vector, label_vector] for seq in batch]
    return:
        [[output_vector]] * batch_size, [[label]]*batch_size
    """

    percentile = 100
    max_len = 32
    pad_index = 0

    lens = [len(dat[0]) for dat in batch]

    # find the max len in each batch
    # dynamical padding
    seq_len = math.ceil(max(lens) * 0.75)
        # or seq_len = max(lens)

    output = []
    out_label = []
    for dat in batch:
        seq = dat[0][:seq_len]
        label = dat[1]

        padding = [pad_index for _ in range(seq_len - len(seq))]

        seq.extend(padding)

        output.append(seq)
        out_label.append(label)

    output = torch.tensor(output, dtype=torch.long)
    out_label = torch.tensor(out_label, dtype=torch.long)

    return output.to('cuda'), out_label.to('cuda')



def load_from_csv(path):
    """
    读取csv格式的数据
    Args:
        path: 文件路径

    Returns:
        {'text': text, 'label':label}
    """
    df = pd.read_csv(path, delimiter=",", names=['text', 'label'])
    data = {'text': df['text'].values, 'label': df['label'].values}

    return data


def sentence_to_ids(params, words:list):
    """
    基于训练集的词典，将句子分词后的结果用数字表示、
    加入unk替换
    Args:
        params:
        words:

    Returns:
        数字形式的tensor向量

    """
    # 将没有出现的词用unk替换
    indices = [params.word2idx.get(w, params.word2idx['unk']) for w in words]
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    return indices_tensor


def build_vocab(params) -> list:
    train_data = load_from_csv(params.train_path)['text']

    vocab_save = params.vocab_save
    if os.path.exists(vocab_save):
        vocab = joblib.load(vocab_save)
    else:
        vocab = {}
        # thread_data = []
        # scale = 1000
        # group = math.ceil(len(train_data) / 1000)
        # for i in range(group):
        #     start = i
        #     end = (start + 1) * scale if (start + 1) * scale < len(train_data) else len(train_data)
        #     thread_data.append(train_data[start:end])
        #
        # for i in range(group):
        #     task = MyThread(sentence_process, args=((sentence for sentence in thread_data[i]), params.tokenizer, params.language))
        #     task.start()
        #     task.join()
        #     words = task.get_result()
        #     for word in words:
        #         vocab.update({word: vocab.get(word, 0) + 1})
        print('sentence processing...')
        for sentence in tqdm(train_data):
            words = list(sentence_process(sentence, params.tokenizer, params.language))
            for word in words:
                vocab.update({word: vocab.get(word, 0) + 1})

        # 筛选freq > 5的 word
        vocab = [word for word, freq in vocab.items() if freq >= params.min_freq]

        vocab.insert(params.unk, 'unk')
        vocab.insert(params.pad, 'pad')

        joblib.dump(vocab, vocab_save)

    print('词典构建完成 ：{} words'.format(len(vocab)))

    return vocab


def load_pretrained_embedding(params) -> numpy.ndarray:
    """
    根据数据集词库的word2idx和raw_word2idx，抽取预训练的embedding信息
    Args:
        params:参数

    Returns:
        返回数据集中单词的embedding_weights信息, 字典类型
    """
    if os.path.exists(params.embedding_save):
        embeddings = np.load(params.embedding_save)
    else:
        embeddings = np.zeros(shape= (params.vocab_size, params.embedding_dim))
        outer_embeddings = np.load(params.raw_embedding_file)

        # 找到vocab中每个元素 在 raw_vocab 中对应的下标
        both_words = params.raw_word2idx.keys() & params.word2idx.keys()
        both_ids = [params.raw_word2idx[i] for i in both_words]
        both_embeddings = outer_embeddings.take(both_ids, axis=0)

        embeddings[:len(both_embeddings),:] = both_embeddings
        differ_words = params.word2idx.keys() - both_words
        differ_ids = [params.word2idx[i] for i in differ_words] # 只在当前语料库有，整体词典没有的word



        # 外部嵌入没有出现的词汇有embedding的平均值代替
        avg = np.average(embeddings, axis=0).reshape(1, 300)
        for idx in differ_ids:
            np.insert(embeddings, idx, avg, axis=0)

        np.save(params.embedding_save, embeddings)

    return embeddings




