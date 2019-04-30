import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sec_2a import split_data

train_X, _, test_X, train_y, _, test_y = split_data(val_ratio=0, split_method=1, random_state=0)

y_probs = {
    'Logistic Regression':
        (LogisticRegression(solver='lbfgs', max_iter=10000)
             .fit(train_X[['NDAI', 'SD', 'CORR']], train_y)
             .predict_proba(test_X[['NDAI', 'SD', 'CORR']])[:, 1]),
    'AdaBoost':
        (AdaBoostClassifier(n_estimators=125, learning_rate=1.25)
             .fit(train_X[['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']], train_y)
             .predict_proba(test_X[['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']])[:, 1]),
    'KNN':
        (Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=35))])
             .fit(train_X[['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']], train_y)
             .predict_proba(test_X[['NDAI', 'SD', 'CORR', 'angle_DF', 'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN']])[:, 1]),
}

df = pd.DataFrame(columns=['fpr', 'tpr', 'thres', 'model'])
for model, y_prob in y_probs.items():
    fpr, tpr, thresholds = roc_curve(test_y, y_prob, pos_label=1)
    df = pd.concat([df, pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thres': thresholds, 'model': model})])

fig, ax = plt.subplots(figsize=(5, 4))
for label, df in df.groupby('model'):
    df.plot(x='fpr', y='tpr', ax=ax, label=label)

    best_thres_idx = (df['tpr'] - df['fpr']).idxmax()
    threshold = df.loc[best_thres_idx, 'thres']
    plt.plot([df.loc[best_thres_idx, 'fpr']], [df.loc[best_thres_idx, 'tpr']],
             marker='o', markersize=5, color="red")

    print('Optimal threshold for', label, '=', threshold)

plt.plot([0, 1], [0, 1], linestyle='dashed')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.xlim([0, 1])
plt.ylim([0, 1.01])
plt.legend()
plt.show()
