
import string
import re
import os

import nltk.stem
from nltk.corpus import stopwords
from zhon.hanzi import punctuation
from torch.utils.data import Dataset
from collections import defaultdict

from Models.BASE import dataset_base


class My_Dataset(dataset_base):
    def __init__(self, config, path, vocab_file):
        """

        Args:
            config: 配置参数
            path: 指定创建数据集的路径
            vocab_file: 通过build_vocab方法创建的语言总词典，区分一元模型和二元模型
        """
        self.config = config
        self.contents = []
        self.labels = []
        self.tokenizer = config.tokenizer
        self.vocab = vocab_file


        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            content = line[0]
            label = line[1]
            self.contents.append(content)
            self.labels.append(label)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        pass




def build_vocab(config, text_list:list):
    """

    Args:
        config: 配置类对象
        text_list: 文本readlines()方法的结果，不经过任何处理

    Returns:
        {(token，词频) : 标号}
            key 是 tuple
            value 是 num

    """
    LM_type = config.LM_type

    tokenizer = config.tokenizer
    min_freq = config.min_freq

    language = config.language
    max_size = config.max_size

    vocab_dic = {}

    # with open(file_path, 'r', encoding='utf8') as f:
    #     lines = f.readlines()

    # 处理文本列表
    print("读取文本...")
    for line in text_list:
        line = line.strip().split('\t')
        if not line:  # 如果是空，直接跳过后续
            continue
        content = line[0]  # 用\t分割，第一个是文本，第二个是类别

        tokens = tokenizer(content)

        # 去停用词 、去除标点、词干抽取（对于英文）
        if LM_type == 'unigram':
            tokens = process_unigram(tokens, language)
        elif LM_type == 'bigram':
            tokens = process_bigram(tokens, language)
        else:
            print('没有匹配的语言模型，无法构建词表')
            exit(2)

        # 生成词表字典，这个字典的get就是有这个元素就返回结果，数量在这里原始值加1; 如果没有返回默认值为0，数量加1
        for word in tokens:
            vocab_dic[word] = vocab_dic.get(word, 0) + 1

    # 首先是根据频次筛选，然后sort降序，然后取词表最大长度；dic_item[1] 表示词频
    vocab_list = sorted([dic_item for dic_item in vocab_dic.items() if dic_item[1] >= min_freq],
                        key=lambda x: x[1], reverse=True)[:max_size]


    # 生成token-序号之间的对应
    vocab_dic = {word: idx for idx, word in enumerate(vocab_list)}


    # 在末尾加上UNK和PAD两个TOKEN
    # vocab_dic.update({config.UNK: len(vocab_dic), config.PAD: len(vocab_dic) + 1})

    print(f"词表构建完成, Vocab size: {len(vocab_dic)}")

    return vocab_dic




def process_bigram(tokens, language):
    """

    将分词后的结果经过处理，生成bigram列表

    Args:
        tokens: 分词后的结果，没有经过任何处理
        language: 语言类型，中文zh或英文en

    Returns:
        返回一个bigram的列表
    """
    BOS, EOS = 'BOS', 'EOS'
    joiner = '|'  # 二元语言模型中的连接符

    # 按照一元语法处理，包括移除标点，移除停用词，stemming操作
    unigram_tokens = process_unigram(tokens, language)

    #  在unigram处理的结果中添加EOS和BOS
    extended_tokens = [BOS]
    extended_tokens.extend(unigram_tokens)
    extended_tokens.append(EOS)

    # 利用unigram_tokens生成bigram_tokens
    bigram_tokens = []

    for i in range(len(extended_tokens) - 1):
        # 条件概率：tokens[i]出现的条件下 tokens[i+1]的概率
        bigram_token = extended_tokens[i + 1] + joiner + extended_tokens[i]
        bigram_tokens.append(bigram_token)

    return bigram_tokens






def process_unigram(tokens, language):
    """
    对分词后的结果去去停用词 、去除标点、词干抽取（对于英文）
    Args:
        tokens: 分词后的结果
        language: 语言类型

    Returns:
        对于中文：移除标点, 去除停用词, 的tokens列表
        对于英文：移除标点, 去停用词, 词干抽取; 后的tokens列表

    """
    # 英文数据集
    if language == 'en':
        # 移除标点
        remove = list(string.punctuation) # 待移除的标点列表
        without_punctuation = [x for x in tokens if x not in remove]

        # 移除停用词
        stopwords_n = []

        #读取自定义的停用词
        with open("../../data/stopwords_en.txt", 'r', encoding='utf8') as f:
            line = f.readline().strip()
            while line:
                stopwords_n.append(line)
                line = f.readline().strip()

        # 移除停用词的操作
        without_stopwords = []
        for w in without_punctuation:
            w = w.lower()
            if w not in stopwords_n:
                without_stopwords.append(w)

        # 词干抽取
        s = nltk.stem.SnowballStemmer('english')  # 参数是选择的语言
        stemmed_tokens = [s.stem(ws.lower()) for ws in without_stopwords]

        return stemmed_tokens

    # 中文数据集
    elif language == 'zh':
        # 移除标点
        remove = list(punctuation)
        without_punctuation = [x for x in tokens if x not in remove]

        # 移除停用词
        stopwords_n = []
        with open("../../data/stopwords_zh.txt", 'r', encoding='utf8') as f:
            line = f.readline().strip()
            while line:
                stopwords_n.append(line)
                line = f.readline().strip()

        without_stopwords = [w for w in without_punctuation
                             if w not in stopwords_n
                             and not re.match(re.compile(r'^\d*.?\d$'), w)]  # 排除所有数字或者评分，如果不匹配则返回none

        return without_stopwords

    # 其他
    else:
        print("没有匹配的语言，请检查数据集")
        exit(1)






def get_label_docs_probs(config):
    """
        得到每个类别的文本集合 和 每个类别的概率
    Args:
        config:

    Returns:
        label_docs:每个类别对应的文本字典
        label_probs:每个类别的概率字典
    """
    train_path = config.train_path

    label_docs = defaultdict(list)
    label_probs = defaultdict(float)

    with open(train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    text_num = len(lines)

    for line in lines:
        line = line.strip().split('\t')
        if len(line) == 2:
            content = line[0]
            label = line[1]
        else:
            continue
        # 将每个label对应的句子加入label_docs
        label_docs[label].append(content)

    # 根据数据集，计算每一个label的概率
    for label_ in label_docs.keys():
        label_docs_num = len(label_docs[label_])

        label_prob = label_docs_num / text_num

        label_probs[label_] += label_prob

    # 计算label_probs

    return label_docs, label_probs


def load_file(filename):
    """
    将文件转换为列表格式
    Args:
        filename: 数据集，目录

    Returns:每行数据构成的列表.去除行尾空格

    """
    f = open(filename, 'r', encoding='utf-8')
    lines = f.readlines()
    res = [x.strip() for x in lines]
    f.close()

    return res



def get_vocab(filepath, config, text_list) -> dict:
    """
        如果filepath存在，就读取filepath的内容。否则config, text_list新建vocab
    Args:
        filepath: 文件路径
        config: 配置
        text_list: 文本列表

    Returns:
        vocab字典，元素都是str类型
    """
    vocab = {}
    if not os.path.exists(filepath):

        vocab = build_vocab(config,text_list)

        with open(filepath, 'w', encoding='utf8') as f:
            for token_freq, idx in vocab.items():
                f.write(str(token_freq) + '\t' + str(idx) + '\n')

        print('语言模型已存入')

    else:
        with open(filepath, 'r', encoding='utf8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.replace(' ','')
            line = line.strip().split('\t')
            token = line[0]
            idx = line[1]
            vocab.update({token: idx})
    return vocab
