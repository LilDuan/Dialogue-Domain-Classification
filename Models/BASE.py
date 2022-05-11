from torch.utils.data import Dataset
import torch
from pyhanlp import *
# HanLP.Args.ShowTermNature = False




class Params_base(object):
    def __init__(self, dataset):
        self.dataset = dataset

        self.datasets = {
            'CrossWOZ': 'zh',
            'RiSAWOZ': 'zh',
            'SGD': 'en'
        }
        self.language = self.datasets[dataset]

        self.dataset_path = os.path.join("../../data", dataset)

        self.train_path = os.path.join("../../data", dataset, "train.csv")
        self.test_path = os.path.join("../../data", dataset, "test.csv")
        self.dev_path = os.path.join("../../data", dataset, "dev.csv")


        # 如果是中文数据集用HanLP（由于默认的中文词向量用Hanlp分词），否则默认用空格分离
        self.tokenizer = lambda x: HanLP.segment(x) if self.language == 'zh' else x.split()

        self.min_freq = 5  # 词表中的最低频率
        # self.max_size = 5000  # 词表中单词的数目上限

        # 类列表 & 每个类的数目
        self.class_list = [x.strip() for x in open(os.path.join(self.dataset_path, "class.txt"), encoding="utf-8").readlines()]
        self.class_num = len(self.class_list)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        # self.device = 'cpu'   # 设备
