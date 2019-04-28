import random

import pandas as pd
import numpy as np


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def load_data(force=False, cache='../data/cache.pkl'):
    def load_image(i):
        with open(f'../data/image{i}.txt') as f:
            lines = list(map(lambda line: line.split(), f.readlines()))
            df = pd.DataFrame(lines, columns=['x', 'y', 'label', 'NDAI', 'SD', 'CORR', 'angle_DF',
                                              'angle_CF', 'angle_BF', 'angle_AF', 'angle_AN'])
            df = df.apply(pd.to_numeric)
            df['source'] = i
            return df

    if not force:
        try:
            return pd.read_pickle(cache)
        except (FileNotFoundError, KeyError):
            pass
    df = pd.concat(list(map(load_image, range(1, 4))), ignore_index=True)
    df.to_pickle(cache)
    return df


if __name__ == '__main__':
    data = load_data(True)
    print(data.shape)
