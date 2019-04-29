import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from common import set_seed
from sec_2a import split_data
from sec_2d import CVgeneric


def run_model(classifier, feature_list=None, K=5):
    result = pd.DataFrame(
        columns=['Features', 'Split Method', 'Test Accuracy', 'Mean CV Accuracy'] + [f'Fold {i}' for i in range(1, K + 1)])

    print(f"Running {classifier.__class__.__name__}")

    for features in feature_list:
        for split_method in [1, 2]:
            train_X, _, test_X, train_y, _, test_y = split_data(val_ratio=0,
                                                                split_method=split_method,
                                                                random_state=0)

            mean_loss, losses = CVgeneric(classifier=classifier,
                                          X=train_X,
                                          y=train_y,
                                          K=5,
                                          scoring=accuracy_score,
                                          split_method=split_method,
                                          features=features,
                                          random_state=1)
            test_loss = accuracy_score(test_y, classifier.predict(test_X[features]))

            result.loc[result.shape[0]] = [','.join(features), split_method, test_loss, mean_loss] + losses
    print()

    return result.applymap(lambda x: "{:.2f}%".format(round(float(x) * 100, 2)) if type(x) is float else x)


if __name__ == '__main__':
    set_seed(0)

    # print(run_model(classifier=LogisticRegression(solver='lbfgs', max_iter=10000),
    #                 feature_list=[['NDAI', 'SD', 'CORR'],
    #                               ['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']
    #                               ]))
    #
    # print(run_model(classifier=AdaBoostClassifier(),
    #                 feature_list=[['NDAI', 'SD', 'CORR'],
    #                               ['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']
    #                               ]))

    print(run_model(classifier=KNeighborsClassifier(),
                    feature_list=[['NDAI', 'SD', 'CORR'],
                                  ['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']
                                  ]))
