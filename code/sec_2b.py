from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from common import set_seed
from sec_2a import split_data


class TrivialClassifier(DummyClassifier):
    """
    A trivial classifier that always predicts a constant label
    """

    def __init__(self, label=-1):
        super().__init__(strategy='constant', constant=label)


if __name__ == '__main__':
    set_seed(0)

    train_X, val_X, test_X, train_y, val_y, test_y = split_data()

    classifier = TrivialClassifier()
    classifier.fit(train_X, train_y)

    print("Trivial classifier accuracy (train) = ", accuracy_score(train_y, classifier.predict(train_X)))
    print("Trivial classifier accuracy (val)   = ", accuracy_score(val_y, classifier.predict(val_X)))
    print("Trivial classifier accuracy (test)  = ", accuracy_score(test_y, classifier.predict(test_X)))
