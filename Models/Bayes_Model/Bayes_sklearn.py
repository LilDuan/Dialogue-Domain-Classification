from Bayes_my import *
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics import recall_score, classification_report


class config_sklearnNB(config_Bayes):
    def __init__(self, dataset):
        super().__init__(dataset)

        self.x_train = [x.strip().split('\t')[0] for x in load_file(self.train_path)]
        self.y_train = [x.strip().split('\t')[1] for x in load_file(self.train_path)]

        self.x_test = [x.strip().split('\t')[0] for x in load_file(self.test_path)]
        self.y_test = [x.strip().split('\t')[1] for x in load_file(self.test_path)]


def bayes_Gaussian(config):
    count_vec = CountVectorizer(tokenizer=config.tokenizer, stop_words=config.stop_words_auto)

    x_count_train = count_vec.fit_transform(config.x_train)
    x_count_test = count_vec.transform(config.x_test)

    GNB = GaussianNB().fit(x_count_train.todense(), config.y_train)

    acc = GNB.score(x_count_test.todense(), config.y_test)

    y_pred = GaussianNB().predict(config.y_test)
    recall = recall_score(config.y_test, y_pred)


    eval_acc = f"{config.dataset} [accuracy] ：{acc}"
    eval_recall = f"{config.dataset} [recall] ：{recall}"

    with open(config.evaluation_path, 'a', encoding='utf-8') as f:
        f.write(eval_acc + '\n')
        f.write(eval_recall + '\n')


def bayes_Multinomial(config):
    count_vec = CountVectorizer(tokenizer=config.tokenizer, stop_words=config.stop_words_auto)

    x_train_count = count_vec.fit_transform(config.x_train)
    x_test_count = count_vec.transform(config.x_test)

    MNB = MultinomialNB()
    MNB.fit(x_train_count.todense(), config.y_train)


    # acc = MNB.score(x_test_count.todense(), config.y_test)
    y_pred_count = MNB.predict(x_test_count)
    #
    # eval_acc = f"数据集{config.dataset}[accuracy]：{acc}"
    # eval_recall = f"{config.dataset}[recall]：{recall}"

    eval = classification_report(config.y_test, y_pred_count, target_names = config.class_list)

    with open(config.evaluation_path, 'a', encoding='utf-8') as f:
        f.write(eval + '\t')

