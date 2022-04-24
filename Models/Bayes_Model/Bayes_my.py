# encoding = utf8

from sklearn.metrics import accuracy_score, precision_score, f1_score
from tqdm import tqdm

from Models.BASE import config_base
from utils import *


class config_Bayes(config_base):

    def __init__(self, dataset):

        super().__init__(dataset)

        self.LM_type = 'bigram'
        self.BOS, self.EOS = "BOS", "EOS"

        self.default_token_prob = 1  # 遇到未注册词时的默认概率
        self.min_token_prob = 0.01  # 忽略概率小于0.01的token
        self.default_zoom_scale = 0.01

        # 暂存预测和真实结果，用于模型评估
        self.pred_path = f'./y_pred/{self.dataset}.txt'
        self.true_path = f'./y-true/{self.dataset}.txt'

        # 用于存放评价指标
        self.evaluation_path = f'./evaluation.txt'

        self.stop_words_zh = [x.strip() for x in load_file(r'../../data/stopwords_zh.txt')]
        self.stop_words_en = [x.strip() for x in load_file(r'../../data/stopwords_en.txt')]
        self.stop_words_auto = self.stop_words_en if self.datasets[dataset] == 'en' else self.stop_words_zh

    def get_LM_path(self, label):
        """
        根据类别得到语言模型文件目录
        Args:
            label: 类别

        Returns:目录地址

        """
        if re.match(r'\d+', label):
            return os.path.join('.\\', 'LM', self.dataset, f'label_{label}.txt')
        else:
            return os.path.join('.\\', 'LM', self.dataset, f'{label}_vocab.txt')


class Model(object):
    def __init__(self, config, vocab_overall):
        """

        Args:
            config:  配置参数
            vocab_overall:  用什么数据构建整体语料库
        """

        self.default_token_prob = config.default_token_prob  # 遇到未注册词时的默认概率
        self.min_token_prob = config.min_token_prob  # 忽略概率小于0.01的token

        self.label_docs_dict, self.label_probs_dict = get_label_docs_probs(config)
        self.tokenizer = config.tokenizer
        self.vocab_overall = vocab_overall  # 基于整个语料库的词典

    def predict(self, config, sentence):
        """

        Args:
            config: 配置
            sentence: 预测句子

        Returns:
            预测结果的类别
        """

        default_token_prob = self.default_token_prob
        min_token_prob = self.min_token_prob
        LM_type = config.LM_type

        label_probs_dict = self.label_probs_dict  # 每个类出现的概率
        label_docs_dict = self.label_docs_dict  # 每个类的文本

        # 存储sentence构建的语言模型
        sentence_tokens = process_bigram(self.tokenizer(sentence), config.language) \
            if LM_type == 'bigram' \
            else process_unigram(self.tokenizer(sentence), config.language)

        likelihood_list = len(sentence_tokens) * [0]  # 存放sentence中每个词的出现概率
        likelihood = 1.0

        vocab_overall = self.vocab_overall

        predictions = defaultdict(float)  # 存储每个类别的预测结果

        for label, docs in label_docs_dict.items():
            # 预存入的语言模型
            LM_file_label = config.get_LM_path(label)
            label_vocab = get_vocab(LM_file_label, config, docs)

            label_tokens_count = 0  # 统计当前类别的总词数

            for doc in docs:
                doc = doc.replace(' ', '')
                if LM_type == 'bigram':
                    # 按照二元语法处理
                    label_tokens = process_bigram(self.tokenizer(doc), config.language)
                else:
                    # 按照一元语法处理
                    label_tokens = process_unigram(self.tokenizer(doc), config.language)

                label_tokens_count += len(label_tokens)

            label_tokens_count *= config.default_zoom_scale

            for idx, token in enumerate(sentence_tokens):
                # 计算likelihood:
                for item in label_vocab.keys():
                    # 读取到的本地文件中str转为tuple
                    if not type(item) == tuple:
                        item = eval(item.split('\t')[0])
                    value = item[0]
                    freq = item[1]  # 这个类别中token出现的次数

                    if value == token:
                        likelihood_list[idx] += (
                            likelihood_formula_laplace(freq, label_tokens_count, len(vocab_overall)))

                        # print(f"\n匹配成功的token: \"{item}\"\t"
                        #       f"当前位于类别 {label}\t"
                        #       f"类别{label}的token_num = {label_tokens_count}\t 当前value的似然概率：{likelihood_list[idx]}", end = '\t')

            # 句子整体的的likelihood
            for prob in likelihood_list:
                likelihood = likelihood * prob if prob > min_token_prob else default_token_prob
            # print(f"\n句子在类别{label}的似然概率是：{likelihood}")

            prediction = bayes_formula(label_probs_dict[label], likelihood)
            # print(f"本轮预测结果：类别{label}，概率：{prediction}")

            predictions[label] = prediction

        predictions_sorted = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        # print(predictions_sorted)

        y = predictions_sorted[0]
        return y[0]

    def evaluate(self, config):
        """

        Args:
            config:

        Returns:
            返回3个参数：准确率，召回率，f1
        """
        test_path = config.test_path
        test_data = load_file(test_path)

        pred_path = config.pred_path
        true_path = config.true_path

        f_pred = open(pred_path, 'r', encoding='utf8')
        f_true = open(true_path, 'r', encoding='utf8')

        y_true, y_pred = [], []
        for line in tqdm(test_data):
            line = line.strip().split('\t')
            content = line[0]
            label = line[1]
            pred = self.predict(config, content)
            y_true.append(label)
            y_pred.append(pred)

            f_true.write(label + '\n')
            f_pred.write(pred + '\n')

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        f_pred.close()
        f_true.close()

        return accuracy, precision, f1


def bayes_formula(prior, _likelihood):
    """
        因为在整个语料库中，句子的概率相同，所以省略分母
    Args:
        prior: 先验概率，即label_i的概率
        _likelihood: 在事件A发生的情况下，evidence的概率有多大；即在当前类别情况下，句子的概率

    Returns: 概率计算结果

    """
    return _likelihood * prior


def likelihood_formula_laplace(word_freq, label_token_count, vocab_len):
    """
    加入Laplace平滑的似然计算
    Args:
        word_freq:  在类别i中单词出现的次数
        label_token_count: 这个类别中的总单词数
        vocab_len: 整体语料库的大小，默认为max_size

    Returns:
        一个单词的概率likelihood
    """
    return (word_freq + 1) / (label_token_count + vocab_len)
