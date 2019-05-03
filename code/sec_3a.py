import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import set_seed
from sec_2a import split_data
from sec_2d import CVgeneric


def run_model(classifier, feature_list=None, K=5):
    """
    Run the given classifier and report cv losses

    :param classifier:   The classifier
    :param feature_list: Subset of features to use
    :param K:            Number of folds in cv
    :return: A summary table
    """
    result = pd.DataFrame(
        columns=['Features', 'Split Method', 'Test Accuracy', 'Mean CV Accuracy'] + [f'Fold {i}' for i in
                                                                                     range(1, K + 1)])

    print(f"Running {classifier}")

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
    # Notice: Due to the extremely long time required to run each model, please only
    # uncomment the part that you want to run.
    set_seed(0)

    # ################################################
    # # 1. Logistic Regression
    # ################################################
    # # Run Model
    # print(run_model(classifier=LogisticRegression(solver='lbfgs', max_iter=10000),
    #                 feature_list=[['NDAI', 'SD', 'CORR'],
    #                               ['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']
    #                               ]))
    #
    # ################################################
    # # 2. AdaBoost
    # ################################################
    # # Optimizing hyper-parameter
    # train_X, _, test_X, train_y, _, test_y = split_data(val_ratio=0, split_method=1, random_state=0)
    # clf = GridSearchCV(AdaBoostClassifier(), param_grid={
    #     'n_estimators': [50, 100, 125, 150],
    #     'learning_rate': [0.5, 1, 1.5, 2]
    # }, cv=5)
    # clf.fit(train_X[['NDAI', 'SD', 'CORR']], train_y)
    # print('Best hyperparameter for AdaBoost', clf.best_params_)
    #
    # # Run Model (with best hyper-parameter)
    # print(run_model(classifier=AdaBoostClassifier(n_estimators=125, learning_rate=1.25),
    #                 feature_list=[['NDAI', 'SD', 'CORR'],
    #                               ['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']
    #                               ]))
    #
    # ################################################
    # # 3. KNN
    # ################################################
    # # Optimizing hyper-parameter
    # train_X, _, test_X, train_y, _, test_y = split_data(val_ratio=0, split_method=1, random_state=0)
    # clf = GridSearchCV(estimator=Pipeline([
    #     ('scale', StandardScaler()),
    #     ('knn', KNeighborsClassifier())
    # ]),
    #     param_grid={
    #         'knn__n_neighbors': [20, 25, 30, 35, 40]
    #     }, cv=5)
    # clf.fit(train_X[['NDAI', 'SD', 'CORR']], train_y)
    # print('Best hyperparameter for KNN', clf.best_params_)
    #
    # # Run Model (with best hyper-parameter)
    # print(run_model(classifier=Pipeline([
    #     ('scale', StandardScaler()),
    #     ('knn', KNeighborsClassifier(n_neighbors=35))
    # ]),
    #     feature_list=[['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']
    #                   ]))
    #
    # ################################################
    # # 4. QDA
    # ################################################
    # # Run Model
    # print(run_model(classifier=QuadraticDiscriminantAnalysis(),
    #                 feature_list=[['NDAI', 'SD', 'CORR'],
    #                               ['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']
    #                               ]))
