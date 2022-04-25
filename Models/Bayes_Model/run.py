import math
import os
import string

import jieba
import re

from nltk.corpus import stopwords
from tqdm import tqdm

from Models.Bayes_Model import Bayes_my, Bayes_my_eval
from Models.Bayes_Model.Bayes_my import config_Bayes

from Models.Bayes_Model.Bayes_sklearn import bayes_Gaussian, config_sklearnNB, bayes_Multinomial
from Models.Bayes_Model.utils import *

if __name__ == '__main__':
    datasets = ['CrossWOZ', 'RiSAWOZ', 'SGD'] # 选择要使用的数据集

    for dataset in datasets:
        print(f"---------当前数据集：{dataset}-----------")
        config = config_Bayes(dataset)
        config_sk = config_sklearnNB(dataset)
        # train_data = load_file(config.train_path)
        # test_data = load_file(config.test_path)
        # dev_data = load_file(config.dev_path)
        #
        # #  构建语言模型
        # vocab_path = config.get_LM_path(dataset)
        # vocab = get_vocab(vocab_path, config, train_data)
        #
        # Bayes_Model = Bayes.Model(config, vocab)
        # eval_bayes.evaluate(config)
        bayes_Multinomial(config_sk)



        # accuracy, precision, f1 = Bayes_Model.evaluate(config)
        # print(f"{dataset}的评估结果:\n "
        #       f"\taccuracy: {accuracy}\n "
        #       f"\tprecision:{precision}\n "
        #       f"\tf1_score:{f1}\n")

        # sentence = '对了，帮我叫下车，从鬼味烤翅到北京华尔顿酒店(原鸿坤国际大酒店)，把车型与车牌告诉我。'
        # # sentence = '人均消费500元。'
        # pred = Bayes_Model.predict(config, sentence)

        # print(pred)



