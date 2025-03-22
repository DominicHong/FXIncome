import numpy as np
import random
from sklearn.linear_model import LogisticRegression


def accuracy(y, prediction):
    tp = 0
    fp = 0
    fn = 0
    for i, (y, pred) in enumerate(zip(y, prediction)):
        if pred == 1 and pred == y:
            tp += 1
        elif pred == 1 and pred != y:
            fp += 1
        elif pred == 0 and pred != y:
            fn += 1
        else:
            continue
    return tp, fp, fn


iteration = 10000
oops = 0
for i in range(iteration):
    X = random.sample(range(1, 100), 20)
    X = np.reshape(X, (len(X), 1))
    y = np.append(np.zeros(3).astype(int), np.ones(17).astype(int))
    clf = LogisticRegression(random_state=0).fit(X, y)
    prediction_0 = (clf.predict_proba(X)[:, 1] >= 0.5).astype(int)
    prediction_1 = (clf.predict_proba(X)[:, 1] >= 0.8).astype(int)
    try:
        tp, fp, fn = accuracy(y, prediction_0)
        precision_0 = tp / (tp + fp)
        recall_0 = tp / (tp + fn)
    except ZeroDivisionError:
        precision_0 = 0
    try:
        tp, fp, fn = accuracy(y, prediction_1)
        precision_1 = tp / (tp + fp)
        recall_1 = tp / (tp + fn)
    except ZeroDivisionError:
        precision_1 = 0
    if precision_1 < precision_0:
        oops += 1
print(f'oops rate when tp > fp :{oops / iteration}')
