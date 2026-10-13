import numpy as np


def one_hot(y, num_classes):
    y = np.asarray(y, dtype=int).reshape(-1)

    vec = np.zeros((len(y), num_classes), dtype=float)
    vec[np.arange(len(y)), y] = 1.0

    return vec


def accuracy(model, X, y):
    correct = 0
    for x, target in zip(X, y):
        pred = model.predict(x)
        if pred == target:
            correct += 1
    return correct / len(X)


def precision(model, X, y):
    true_positive = 0
    false_positive = 0
    for x, target in zip(X, y):
        pred = model.predict(x)
        if pred == target:
            true_positive += 1
        else:
            false_positive += 1
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0


def recall(model, X, y):
    true_positive = 0
    false_negative = 0
    for x, target in zip(X, y):
        pred = model.predict(x)
        if pred == target:
            true_positive += 1
        else:
            false_negative += 1
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0


def f1_score(model, X, y):
    p = precision(model, X, y)
    r = recall(model, X, y)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0


def confusion_matrix(model, X, y):
    num_classes = len(set(y))
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for x, target in zip(X, y):
        pred = model.predict(x)
        matrix[target][pred] += 1

    return matrix


__all__ = ['one_hot', 'accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']