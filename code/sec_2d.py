from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from common import set_seed
from sec_2a import split_data
from sec_2b import TrivialClassifier


def CVgeneric(classifier, X, y, K=3, scoring=accuracy_score):
    """
    A generic cross validation (CV) function that takes a generic classifier,
    training features, training labels, number of folds and a loss function as
    inputs and outputs the K-fold CV loss on the training set.

    :param classifier: The classifier (should has implemented fit() and predict())
    :param X:          Training features
    :param y:          Training labels
    :param K:          Number of folds
    :param scoring:    A callable loss function: scoring(y_true, y_pred)

    :return: K-fold CV loss
    """
    kf = KFold(n_splits=K)
    scores = []
    for train_index, test_index in kf.split(X):
        train_X, test_X = X.iloc[train_index, :], X.iloc[test_index, :]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]
        classifier.fit(train_X, train_y)
        scores.append(scoring(test_y, classifier.predict(test_X)))
    return sum(scores) / K


if __name__ == '__main__':
    set_seed(0)

    train_X, val_X, test_X, train_y, val_y, test_y = split_data()

    K = 3
    trivial_classifier = TrivialClassifier()
    loss = CVgeneric(trivial_classifier, train_X, train_y, K=K, scoring=accuracy_score)

    print(f"{K}-fold CV loss of {trivial_classifier.__class__.__name__} = ", loss)
