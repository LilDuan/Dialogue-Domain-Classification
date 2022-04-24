import jieba
import nltk
from torch.utils.data import Dataset
import os


class config_base(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_path = os.path.join("../../data", dataset, "train.txt")
        self.test_path = os.path.join("../../data", dataset, "test.txt")
        self.dev_path = os.path.join("../../data", dataset, "dev.txt")

        self.datasets = {
            'CrossWOZ': 'zh',
            'RiSAWOZ': 'zh',
            'SGD': 'en'
        }
        self.language = self.datasets[dataset]

        # 如果是中文数据集用jieba，否则用nltk.word_tokenize
        self.tokenizer = lambda sentence: jieba.cut(sentence) if self.language == 'zh' else nltk.word_tokenize(sentence)

        self.min_freq = 2  # 词表中的最低频率
        self.max_size = 1000  # 词表中单词的数目上限

        # 类列表 & 每个类的数目
        self.class_list = [x.strip() for x in open(os.path.join("../../data", dataset, "class.txt"), encoding="utf-8").readlines()]
        self.class_num = len(self.class_list)



class dataset_base(Dataset):
    pass