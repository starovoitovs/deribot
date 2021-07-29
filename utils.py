import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_keys(depth):

    apcs = [f'ap_{i}' for i in range(depth)]
    avcs = [f'av_{i}' for i in range(depth)]
    bpcs = [f'bp_{i}' for i in range(depth)]
    bvcs = [f'bv_{i}' for i in range(depth)]

    keys = [x for items in zip(apcs, bpcs, avcs, bvcs) for x in items]

    return keys, apcs, avcs, bpcs, bvcs


def add_pcolor_to_plot(ax, data):
    
    y0, y1 = ax.get_ylim()
    height = (y1 - y0) * 0.1
    
    X = np.repeat(ax.lines[0].get_xdata().reshape(1, -1), 2, axis=0)
    Y = np.repeat([[0, height]], len(data), axis=0).T + y0 - 2 * height

    ax.pcolor(X, Y, data.to_numpy().reshape(1, -1), cmap='coolwarm', alpha=1., vmin=-1., vmax=1.)


def splitX(df, features, offset=1, window_size=100, depth=10):

    keys = get_keys(depth)[0]

    I = np.array([x + np.arange(window_size) for x in np.arange(0, len(df) - window_size, offset)])

    # X1
    X1 = df[keys].to_numpy()[I]

    bid, ask = X1[:, -1, [0, 1]].T
    mid_price = (bid + ask) / 2

    X1[:, :, [x for x in np.arange(40) if x % 4 < 2]] -= mid_price[:, np.newaxis, np.newaxis]
    X1 = X1[:, :, :, np.newaxis]

    # X2
    X2 = df[features].to_numpy()[I]

    return X1, X2


def splitY(df, offset=1, window_size=100, label_threshold=5e-5):

    y = df.iloc[offset - 1 + window_size:len(df):offset]['y2'].to_numpy()
    y = -1 + 1. * (y >= -label_threshold) + 1. * (y >= label_threshold)

    # one-hot encoder
    enc = OneHotEncoder(sparse=False)
    y = enc.fit_transform(y.reshape(-1, 1))

    return y


def prepare_features(df, depth=10):

    keys, apcs, avcs, bpcs, bvcs = get_keys(depth)

    # diff time and asset price with lag=1 to remove trend
    columns = {
        'timestamp': 'timestamp_diff',
    }
    
    df0 = df[['timestamp']].diff(1).rename(columns=columns)
    df0['timestamp_diff'] /= 1000.

    df1 = df[keys]
    retval_df = pd.concat([df1, df0], axis=1)

    # remove nans
    retval_df = retval_df[~retval_df.isna().any(axis=1)]

    # rescale volumes
    l = 10.
    u = 1000000.
    retval_df[avcs + bvcs] = (retval_df[avcs + bvcs] - l) / (u - l)

    # midprice
    retval_df['mid_price'] = (retval_df['bp_0'] + retval_df['ap_0']) / 2

    return retval_df

    
def prepare_labels(df, kernel_size):
    
    retval_df = df.copy()

    # https://arxiv.org/pdf/1808.03668.pdf, p. 4
    retval_df['m+'] = retval_df['mid_price'].shift(-(kernel_size - 1)).rolling(kernel_size).mean()
    retval_df['m-'] = retval_df['mid_price'].rolling(kernel_size).mean()
    retval_df['y1'] = (retval_df['m+'] - retval_df['mid_price']) / retval_df['mid_price']
    retval_df['y2'] = (retval_df['m+'] - retval_df['m-']) / retval_df['m-']

    retval_df = retval_df.iloc[kernel_size - 1:-(kernel_size - 1)]

    return retval_df

