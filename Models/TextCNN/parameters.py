import numpy as np

from Models.BASE import Params_base
from dataset import *


class Params(Params_base):
    def __init__(self, dataset):
        super().__init__(dataset)

        # hyper-parameters
        self.learning_rate = 1e-3       # 学习率
        self.dropout = 0.2              # 随机失活
        self.num_epochs = 20            # epoch数
        self.batch_size = 128           # mini-batch大小
        self.filter_sizes = (2, 3, 4)   # 卷积核尺寸
        self.num_filters = 256          # 每个尺寸的卷积核的数量
        self.require_improvement = 1000 # 如果超过1000batch还有更新 则自动停止训练


        # self.pad_size = 32  # 每句话处理成的长度(短填长切)

        # 保存模型
        self.model_name = 'TextCNN'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name




        # 词典，embedding信息
        self.embedding_dim = 300  # 字向量维度
        self.min_freq = 5
        self.pad = 0
        self.unk = 1

        # 路径信息
        self.raw_vocab_file = r'../../embedding/vocab_matrix/vocab_{}.npy'.format(self.language)
        self.raw_embedding_file = r'../../embedding/vocab_matrix/matrix_{}.npy'.format(self.language)
        self.vocab_save = r'../../data/vocab/{}.npy'.format(self.dataset)
        self.embedding_save = r'../../embedding/custom_corpus/{}.npy'.format(self.dataset)


        self.raw_vocab = np.load(self.raw_vocab_file, allow_pickle=True).item()
        self.raw_word2idx = self.raw_vocab['w2i']  # 字典类型
        self.raw_idx2word = self.raw_vocab['i2w']  # 列表类型

        list2dict = lambda x: dict(zip(x, range(len(x))))

        self.idx2word = build_vocab(self)
        self.word2idx = list2dict(self.idx2word)

        self.label2name = list2dict(self.class_list)

        self.vocab_size = len(self.idx2word)  # 外部使用之前需要先更新值
