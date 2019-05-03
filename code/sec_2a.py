import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import load_data, justify


def split_data(df=None, split_method=1,
               val_ratio=0.2, test_ratio=0.2,
               keep_unlabeled=False, random_state=None):
    """
    Split the data into test, validation and test sets. Note that validation set is created
    after test set, which means split_data(df, val_ratio=0) and split_data(df, val_ratio=0.2)
    will return the same test set, independent of the creation of validation set.

    The y label should be contained in column named label.

    :param df:             Data source (DataFrame)
    :param split_method:   Which splitting method to use (either 1 or 2)
    :param val_ratio:      Percentage of validation data
    :param test_ratio:     Percentage of test data
    :param keep_unlabeled: Whether to keep unlabeled data
    :param random_state:   Random seed to be used

    :return: (train_X, val_X, test_X, train_y, val_y, test_y)
    """
    ###########################################
    # Initialization
    ###########################################
    # Check if the two ratios are valid
    assert val_ratio + test_ratio <= 1
    assert test_ratio > 0
    assert val_ratio >= 0

    # Check if method is either 1 or 2
    assert split_method in [1, 2], "Method number should be either 1 or 2"

    # Load data from source if not provided
    if df is None:
        df = load_data()

    # Check if y label is contained in dataframe
    assert df.columns.contains('label')

    # Remove unlabeled data if specified
    if keep_unlabeled is False:
        df = df.loc[df['label'] != 0]

    # Initialize all three sets to empty dataframe
    train, val, test = (pd.DataFrame(columns=df.columns),
                        pd.DataFrame(columns=df.columns),
                        pd.DataFrame(columns=df.columns))

    ###########################################
    # Method 1: Split within each label
    ###########################################
    if split_method == 1:
        # Split with fixed ratio within each class
        for label in df['label'].unique():
            cur_train = df.loc[df['label'] == label]
            cur_train, cur_test, _, _ = \
                train_test_split(cur_train, cur_train['label'],
                                 test_size=test_ratio, random_state=random_state)
            cur_train, cur_val, _, _ = \
                train_test_split(cur_train, cur_train['label'],
                                 test_size=val_ratio / (1 - test_ratio), random_state=random_state)

            train = pd.concat([train, cur_train])
            val = pd.concat([val, cur_val])
            test = pd.concat([test, cur_test])

    ###########################################
    # Method 2: Split by down-sample pixels
    ###########################################
    elif split_method == 2:
        def down_sample_pixels(df, ratio, random_state=None):
            # Convert into 2D matrix where column and row indices are x and y coordinates
            # and values are the indices of observations in the original dataframe
            pixels = pd.pivot_table(df.reset_index(), index='x', columns='y', values='index')
            pixels[:] = justify(pixels.values, invalid_val=np.nan, axis=1, side='left')

            x_gap = 1 / np.sqrt(ratio) if ratio > 0 else pixels.shape[0] + 1
            y_gap = 1 / np.sqrt(ratio) if ratio > 0 else pixels.shape[1] + 1

            x_offset = random_state % x_gap if random_state else 0
            y_offset = random_state % y_gap if random_state else 0

            sample_x_idx = list(filter(lambda x: 0 <= x < pixels.shape[0],
                                       x_offset +
                                       np.rint(np.array(range(int(np.floor(pixels.shape[0] / x_gap)))) * x_gap)))
            sample_y_idx = list(filter(lambda y: 0 <= y < pixels.shape[1],
                                       y_offset +
                                       np.rint(np.array(range(int(np.floor(pixels.shape[1] / y_gap)))) * y_gap)))

            sample_idx = pixels.iloc[sample_x_idx, sample_y_idx].values.flatten()
            sample_idx = sample_idx[np.isfinite(sample_idx)]

            return df.loc[~df.index.isin(sample_idx)], df.loc[sample_idx]

        # Down sample pixels from each images
        for source in df['source'].unique():
            cur_train = df.loc[df['source'] == source]
            cur_train, cur_test = down_sample_pixels(cur_train, ratio=test_ratio,
                                                     random_state=random_state)
            cur_train, cur_val = down_sample_pixels(cur_train, ratio=val_ratio / (1 - test_ratio),
                                                    random_state=random_state)
            train = pd.concat([train, cur_train])
            val = pd.concat([val, cur_val])
            test = pd.concat([test, cur_test])

    ###########################################
    # Clean up, Merge, Shuffle
    ###########################################
    # Shuffle data before return
    train, val, test = train.sample(frac=1), val.sample(frac=1) if not val.empty else val, test.sample(frac=1)

    # Split features and label
    train_X, train_y = train.loc[:, ~train.columns.isin(['label'])], train['label']
    val_X, val_y = val.loc[:, ~val.columns.isin(['label'])], val['label']
    test_X, test_y = test.loc[:, ~test.columns.isin(['label'])], test['label']

    return train_X, val_X, test_X, train_y, val_y, test_y


if __name__ == '__main__':
    data = load_data()
    train_X, val_X, test_X, train_y, val_y, test_y = split_data(data, split_method=1, random_state=0)

    total_size = sum([x.shape[0] for x in [train_X, val_X, test_X]])
    print(f"Train size:      X={train_X.shape}, y={train_y.shape} [ratio={train_X.shape[0] / total_size}]")
    print(f"Validation size: X={val_X.shape}, y={val_y.shape} [ratio={val_X.shape[0] / total_size}]")
    print(f"Test size:       X={test_X.shape}, y={test_y.shape} [ratio={test_X.shape[0] / total_size}]")
