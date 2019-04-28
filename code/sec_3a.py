from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sec_2a import split_data
from sec_2d import CVgeneric

if __name__ == '__main__':
    K = 5
    split_method = 1

    # Load and split data
    train_X, _, test_X, train_y, _, test_y = split_data(val_ratio=0,
                                                        split_method=split_method,
                                                        random_state=0)

    # Logistic classifier
    # classifier = LogisticRegression(solver='lbfgs', max_iter=10000)
    # mean_loss, losses = CVgeneric(classifier=classifier,
    #                               X=train_X,
    #                               y=train_y,
    #                               K=5,
    #                               scoring=accuracy_score,
    #                               split_method=split_method,
    #                               random_state=0)
    # print(f"Mean {len(losses)}-fold CV loss of {classifier.__class__.__name__} =", mean_loss)

    # Logistic classifier
    classifier = AdaBoostClassifier()
    mean_loss, losses = CVgeneric(classifier=classifier,
                                  X=train_X,
                                  y=train_y,
                                  K=5,
                                  scoring=accuracy_score,
                                  split_method=split_method,
                                  random_state=0)
    print(f"Mean {len(losses)}-fold CV loss of {classifier.__class__.__name__} =", mean_loss)


