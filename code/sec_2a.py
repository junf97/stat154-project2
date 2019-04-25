import pandas as pd
from sklearn.model_selection import train_test_split

from common import load_data, set_seed


def split_data(df=None, method=1, val_ratio=0.2, test_ratio=0.2):
    """
    Split the data into test, validation and test sets

    :param df:         Data source (DataFrame)
    :param method:     Which splitting method to use (either 1 or 2)
    :param val_ratio:  Percentage of validation data
    :param test_ratio: Percentage of test data

    :return: (train_X, val_X, test_X, train_y, val_y, test_y)
    """
    # Load data from source if not provided
    if df is None:
        df = load_data()

    # Initialize all three sets to empty dataframe
    train, val, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ######################################
    # Method 1: Split within each label
    ######################################
    if method == 1:
        for label in df['label'].unique():
            cur_train = df.loc[df['label'] == label]
            cur_train, cur_val, _, _ = \
                train_test_split(cur_train, cur_train['label'], test_size=val_ratio)
            cur_train, cur_test, _, _ = \
                train_test_split(cur_train, cur_train['label'], test_size=test_ratio / (1 - val_ratio))

            train = pd.concat([train, cur_train])
            val = pd.concat([val, cur_val])
            test = pd.concat([test, cur_test])

    ######################################
    # Method 2:
    ######################################
    elif method == 2:
        pass
    else:
        raise Exception("Method number should be either 1 or 2")

    # Shuffle data before return
    train, val, test = train.sample(frac=1), val.sample(frac=1), test.sample(frac=1)

    # Split features and label
    train_X, train_y = train.loc[:, train.columns != 'label'], train['label']
    val_X, val_y = val.loc[:, val.columns != 'label'], val['label']
    test_X, test_y = test.loc[:, test.columns != 'label'], test['label']

    return train_X, val_X, test_X, train_y, val_y, test_y


if __name__ == '__main__':
    set_seed(0)

    train_X, val_X, test_X, train_y, val_y, test_y = split_data()
    print(f"Train size:      X={train_X.shape}, y={train_y.shape}")
    print(f"Validation size: X={val_X.shape}, y={val_y.shape}")
    print(f"Test size:       X={test_X.shape}, y={test_y.shape}")
