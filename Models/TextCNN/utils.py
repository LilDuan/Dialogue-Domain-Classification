import time
from datetime import timedelta

import nltk
import pandas as pd
import os
from string import punctuation
from pyhanlp import *
HanLP.Config.ShowTermNature = False


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))




def sentence_process(sentence, tokenizer, language):
    """
    包括分词，去停用词 、去除标点、词干抽取（对于英文）
    Args:
        sentence:
        tokenizer:
        language:

    Returns:
        待放入词典的单词列表
    """

    stopwords = [x.strip() for x in open(r'..\..\data\stopwords_{}.txt'.format(language),
                                 'r', encoding='utf-8').readlines()]

    tokens = tokenizer(sentence)

    def lemmatize(words):
        """
        还原单词为默认形态
        Args:
            words: 单词列表

        Returns:
            每个单词还原后的单词列表
        """
        try:
            wnl = nltk.WordNetLemmatizer()
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            wnl = nltk.WordNetLemmatizer()
        return [wnl.lemmatize(t) for t in words]

    without_stopwords = [str(token.word) for token in tokens if str(token.word) not in stopwords]
    if language == 'en': # 去除标点, 还原词形
        without_stopwords = [token for token in without_stopwords if token not in punctuation]
        without_stopwords = lemmatize(without_stopwords)

    return without_stopwords



def txt_to_csv(dataset_path):
    files = os.listdir(dataset_path)
    for file in files:
        if file.__contains__('class'):
            continue
        path = os.path.join(dataset_path, file)
        csv_fname = path.replace('txt', 'csv')

        df = pd.read_csv(path, delimiter="\t")
        df.to_csv(csv_fname, encoding='utf-8', index=False)
    print('{}下csv文件写入完毕！'.format(dataset_path))


"""
利用torchText构建数据集
"""

# def DataLoader(config):
#     TEXT = data.Field(sequential=True,
#                       lower=True,
#                       fix_length=config.pad_size,
#                       tokenize=config.tokenizer,
#                       include_lengths=True,
#                       preprocessing=preProcess,
#                       )
#     LABEL = data.Field(sequential=False,
#                        use_vocab=False
#                        )
#
#     fields = [('text', TEXT), ('label', LABEL)]
#
#     def txt_to_csv(dataset_path):
#         files = os.listdir(dataset_path)
#         for file in files:
#             if file.__contains__('class'):
#                 continue
#             path = os.path.abspath(file)
#             csv_fname = path.replace('txt', 'csv')
#             df = pd.read_csv(path, delimiter="\t")
#             df.to_csv(csv_fname, encoding='utf-8', index=False)
#         print('{}下csv文件写入完毕！'.format(dataset_path))
#
#     txt_to_csv(config.dataset_path)
#
#     train_data, valid_data, test_data = data.TabularDataset.splits(path=config.dataset_path,
#                                                                    format='csv',
#                                                                    train='train.csv',
#                                                                    validation='dev.csv',
#                                                                    test='test.csv',
#                                                                    fields=fields,
#                                                                    skip_header=False
#                                                                    )
#     return TEXT, LABEL, train_data
