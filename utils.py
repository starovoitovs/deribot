import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


apcs = [f'ap_{i}' for i in range(10)]
avcs = [f'av_{i}' for i in range(10)]
bpcs = [f'bp_{i}' for i in range(10)]
bvcs = [f'bv_{i}' for i in range(10)]

keys = [x for items in zip(apcs, bpcs, avcs, bvcs) for x in items]


def add_pcolor_to_plot(ax, data):
    
    y0, y1 = ax.get_ylim()
    height = (y1 - y0) * 0.1
    
    X = np.repeat(ax.lines[0].get_xdata().reshape(1, -1), 2, axis=0)
    Y = np.repeat([[0, height]], len(data), axis=0).T + y0 - 2 * height

    ax.pcolor(X, Y, data.to_numpy().reshape(1, -1), cmap='coolwarm', alpha=1., vmin=-1., vmax=1.)

    
def prepare_df1(df):

    # diff time and asset price with lag=1 to remove trend
    columns = {
        'timestamp': 'timestamp_diff',
        'best_bid_price': f'bid_diff_feature_1',
        'best_ask_price': f'ask_diff_feature_1'
    }
    
    df0 = df[['timestamp', 'best_bid_price', 'best_ask_price']].diff(1).rename(columns=columns)
    df0['timestamp_diff'] /= 1000.

    df1 = df[['best_bid_price', 'best_ask_price'] + keys]
    dfX = pd.concat([df1, df0], axis=1)

    # set prices as relative differences
    dfX[apcs] = dfX[apcs].sub(dfX['ap_0'], axis=0)
    dfX[bpcs] = -dfX[bpcs].sub(dfX['bp_0'], axis=0)

    # remove nans
    dfX = dfX[~dfX.isna().any(axis=1)]

    # rescale volumes
    l = 10.
    u = 1000000.
    dfX[avcs + bvcs] = (dfX[avcs + bvcs] - l) / (u - l)

    # @todo note this is mid_price_diff, either rename or check
    dfX['mid_price'] = (dfX['best_bid_price'] + dfX['best_ask_price']) / 2
    dfX['spread'] = dfX['best_ask_price'] - dfX['best_bid_price']

    return dfX

    
def prepare_df2(df, k):
    
    dfX = df.copy()
    
    # https://arxiv.org/pdf/1808.03668.pdf, p. 4
    dfX['m+'] = dfX['mid_price'].shift(-(k-1)).rolling(k).mean()
    dfX['m-'] = dfX['mid_price'].rolling(k).mean()
    dfX['y1'] = (dfX['m+'] - dfX['mid_price']) / dfX['mid_price']
    dfX['y2'] = (dfX['m+'] - dfX['m-']) / dfX['m-']

    dfX = dfX.iloc[k-1:-(k-1)]

    return dfX

