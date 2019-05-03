import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from sec_2a import split_data
from sec_2b import TrivialClassifier


def CVgeneric(classifier, X, y, K=3, scoring=accuracy_score, split_method=1, features=None, random_state=None):
    """
    A generic cross validation (CV) function that takes a generic classifier,
    training features, training labels, number of folds and a loss function as
    inputs and outputs the K-fold CV loss on the training set.

    :param classifier:    The classifier (should has implemented fit() and predict())
    :param X:             Training features
    :param y:             Training labels
    :param K:             Number of folds
    :param scoring:       A callable loss function: scoring(y_true, y_pred)
    :param split_method:  Method of splitting data (either 1 or 2)
    :param features:      Specify set of features to be used in classification models
    :param random_state:  Random seed to be used

    :return: A tuple: (mean loss, [fold 1 loss, ..., fold K loss])
    """
    assert type(K) is int and K >= 2

    # Create K test sets by using split method defined in section 2a
    test_sets = []
    remain_X, remain_y = X.copy(), y.copy()
    for fold_id in range(K - 1):
        remain = remain_X.merge(remain_y, left_index=True, right_index=True)
        remain_X, _, fold_test_X, remain_y, _, fold_test_y = \
            split_data(remain, split_method, val_ratio=0, test_ratio=1 / (K - fold_id),
                       keep_unlabeled=True, random_state=random_state)
        test_sets.append((fold_test_X, fold_test_y))
    test_sets.append((remain_X, remain_y))

    # Compute loss in each fold
    fold_losses = []
    for fold_id in range(K):
        fold_train_X, fold_train_y = map(pd.concat, zip(*test_sets[:fold_id] + test_sets[fold_id + 1:]))
        fold_test_X, fold_test_y = test_sets[fold_id]

        # Select features if specified, otherwise use all features
        if features is not None:
            fold_train_X = fold_train_X[features]
            fold_test_X = fold_test_X[features]

        # Fit model
        classifier.fit(fold_train_X, fold_train_y)
        fold_losses.append(scoring(fold_test_y, classifier.predict(fold_test_X)))

    return np.mean(fold_losses), fold_losses


if __name__ == '__main__':
    # Load and split data
    train_X, _, test_X, train_y, _, test_y = split_data(val_ratio=0,
                                                        split_method=1,
                                                        random_state=0)

    # Test CVgeneric() on multiple classifier
    for classifier in [TrivialClassifier(),
                       SGDClassifier(max_iter=1000, tol=1e-3)]:
        mean_loss, losses = CVgeneric(classifier=classifier,
                                      X=train_X,
                                      y=train_y,
                                      K=5,
                                      scoring=accuracy_score,
                                      split_method=1,
                                      random_state=0)
        print(f"Mean {len(losses)}-fold CV loss of {classifier.__class__.__name__} =", mean_loss)
