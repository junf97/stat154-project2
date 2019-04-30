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
        except (FileNotFoundError, KeyError, ModuleNotFoundError):
            pass
    df = pd.concat(list(map(load_image, range(1, 4))), ignore_index=True)
    df.to_pickle(cache)
    return df


def justify(a, invalid_val=0, axis=1, side='left'):
    """
    Justifies a 2D array

    Parameters
    ----------
    A : ndarray
        Input array to be justified
    axis : int
        Axis along which justification is to be made
    side : str
        Direction of justification. It could be 'left', 'right', 'up', 'down'
        It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.

    """

    if invalid_val is np.nan:
        mask = ~np.isnan(a)
    else:
        mask = a != invalid_val
    justified_mask = np.sort(mask, axis=axis)
    if (side == 'up') | (side == 'left'):
        justified_mask = np.flip(justified_mask, axis=axis)
    out = np.full(a.shape, invalid_val)
    if axis == 1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out


if __name__ == '__main__':
    data = load_data(True)
    print(data.shape)
