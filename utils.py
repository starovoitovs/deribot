import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_keys(depth):

    bpcs = [f'bp_{i}' for i in range(depth)]
    apcs = [f'ap_{i}' for i in range(depth)]
    bvcs = [f'bv_{i}' for i in range(depth)]
    avcs = [f'av_{i}' for i in range(depth)]

    keys = [x for items in zip(apcs, bpcs, avcs, bvcs) for x in items]

    return keys, bpcs, apcs, bvcs, avcs


def add_pcolor_to_plot(ax, data):
    
    y0, y1 = ax.get_ylim()
    height = (y1 - y0) * 0.1
    
    X = np.repeat(ax.lines[0].get_xdata().reshape(1, -1), 2, axis=0)
    Y = np.repeat([[0, height]], len(data), axis=0).T + y0 - 2 * height

    ax.pcolor(X, Y, data.to_numpy().reshape(1, -1), cmap='coolwarm', alpha=1., vmin=-1., vmax=1.)


def split_x(df, offset=1, window_size=100, depth=10):

    keys = get_keys(depth)[0]

    I = np.array([x + np.arange(window_size) for x in np.arange(0, len(df) - window_size, offset)])

    # X1
    X1 = df[keys].to_numpy()[I]

    bid, ask = X1[:, -1, [0, 1]].T
    mid_price = (bid + ask) / 2

    # shift prices
    pmask = [x for x in np.arange(4 * depth) if x % 4 < 2]
    X1[:, :, pmask] -= mid_price[:, np.newaxis, np.newaxis]

    # add axis
    X1 = X1[:, :, :, np.newaxis]

    # X2
    timestamp_diff = df[['timestamp']].diff(1) / 1000.
    X2 = timestamp_diff.to_numpy()[I]

    return X1, X2


def split_y(df, offset=1, window_size=100, label_threshold=5e-5):

    y = df.iloc[offset - 1 + window_size:len(df):offset]['y2'].to_numpy()
    y = -1 + 1. * (y >= -label_threshold) + 1. * (y >= label_threshold)

    # one-hot encoder
    enc = OneHotEncoder(sparse=False)
    y = enc.fit_transform(y.reshape(-1, 1))

    return y


def prepare_features(df, depth=10):

    # express volumes in the units of contracts (originally in USD)
    keys, bpcs, apcs, bvcs, avcs = get_keys(depth)
    df.loc[:, avcs + bvcs] = df.loc[:, avcs + bvcs].div(df['index_price'], axis=0)

    return df


def prepare_labels(df, kernel_size):
    
    retval_df = df.copy()

    mid_price = (retval_df['ap_0'] + retval_df['bp_0']) / 2

    # https://arxiv.org/pdf/1808.03668.pdf, p. 4
    retval_df['m+'] = mid_price.shift(-(kernel_size - 1)).rolling(kernel_size).mean()
    retval_df['m-'] = mid_price.rolling(kernel_size).mean()
    retval_df['y1'] = (retval_df['m+'] - mid_price) / mid_price
    retval_df['y2'] = (retval_df['m+'] - retval_df['m-']) / retval_df['m-']

    retval_df = retval_df.iloc[kernel_size - 1:-(kernel_size - 1)]

    return retval_df

