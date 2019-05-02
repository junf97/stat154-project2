import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sec_2a import split_data

train_X, _, test_X, train_y, _, test_y = split_data(val_ratio=0, split_method=1, random_state=0)

y_predicts = {
    'Logistic Regression':
        pd.Series(LogisticRegression(solver='lbfgs', max_iter=10000)
             .fit(train_X[['NDAI', 'SD', 'CORR']], train_y)
             .predict_proba(test_X[['NDAI', 'SD', 'CORR']])[:, 1] >= 0.2940451376607227)
             .apply(lambda b: 1 if b else -1),
    'AdaBoost':
        pd.Series(AdaBoostClassifier(n_estimators=125, learning_rate=1.25)
             .fit(train_X[['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']], train_y)
             .predict_proba(test_X[['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']])[:, 1] >= 0.498436455140852)
             .apply(lambda b: 1 if b else -1),
    'KNN':
        pd.Series(Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=35))])
             .fit(train_X[['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']], train_y)
             .predict_proba(test_X[['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']])[:, 1] >= 0.37142857142857144)
             .apply(lambda b: 1 if b else -1),
    'QDA':
        pd.Series(QuadraticDiscriminantAnalysis()
             .fit(train_X[['NDAI', 'SD', 'CORR']], train_y)
             .predict_proba(test_X[['NDAI', 'SD', 'CORR']])[:, 1] >= 0.15130378416778748)
             .apply(lambda b: 1 if b else -1),
}

for model, y_predict in y_predicts.items():
    print(model, 'Classification Report:')
    print(classification_report(test_y, y_predict, target_names=['Cloudless', 'Cloudy']))
    print()
