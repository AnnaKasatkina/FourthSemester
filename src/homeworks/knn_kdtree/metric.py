import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred):
    unique_classes = np.unique(y_true)
    f1_scores = []

    for cls in unique_classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_scores.append(f1)

    return np.mean(f1_scores)
