#-*- coding: UTF-8-*-


# from pyhanlp import *
# HanLP.Config.ShowTermNature = False
#
# tokenizer = lambda x: HanLP.segment(x) if text.__contains__('zh') else x.split()
#
# text = 'zh游玩结束后我想找一个  评分4.5分以上的餐馆吃饭，给我推荐一家吧。'
# print(tokenizer(text))
# import os
# import re
#
# import nltk
# import numpy as np
# import pandas as pd
import csv
# import TextCNN
# from Models.TextCNN import parameters
# from Models.TextCNN.dataset import *
# from nltk.stem import WordNetLemmatizer

#
# path = r'D:\AI\Graduation Project\data\CrossWOZ\train.txt'
# out_filename = r'D:\AI\Graduation Project\data\CrossWOZ\train.csv'
# csv_fname = 'text.csv'
# df = pd.read_csv(path, delimiter="\t")
#
# df.to_csv(csv_fname, encoding='utf-8', index=False)
#
# data = pd.read_csv(csv_fname, delimiter=",", names=['text','label'])
#
# reverse = lambda x: dict(zip(x, range(len(x))))
# print(reverse(['word','unk','pad']))


# datasets = ['CrossWOZ', 'RiSAWOZ', 'SGD']
# for dataset in datasets:
#     args = TextCNN.Args(dataset)
#     txt_to_csv(args.dataset_path)
#
# # #
# params = parameters.Params('CrossWOZ')
# train_data, dev_data, test_data = data_train_dev_test(params)
# vocabInstance = Vocab(params, train_data)


# path = _vocab.vocab_path_raw
# data = np.load(path,allow_pickle=True)
# dat = np.expand_dims(data, 0)


import torch
import torch.nn as nn
import numpy as np
from torch import tensor

a = np.zeros(shape=(2,3))
print(a)

b = np.ones((1,3))
print(b)

a[:1,:] = b
print(a)






