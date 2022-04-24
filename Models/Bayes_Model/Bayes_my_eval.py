import Bayes_my
from Models.Bayes_Model.utils import load_file

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score



def evaluate(config):
    pred_path = config.pred_path
    true_path = config.true_path

    y_true = load_file(true_path)
    y_pred = load_file(pred_path)


    acc = metrics.accuracy_score(y_true,y_pred)
    print(acc)
