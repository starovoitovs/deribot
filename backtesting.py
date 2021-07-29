import numpy as np

from utils import splitX


def integrate(df, strategy_column='strategy', bid_column='best_bid_price', ask_column='best_ask_price'):

    df = df[[bid_column, ask_column, strategy_column]]
    df = df[~df.isna().any(axis=1)]

    strategy_diff = df[strategy_column].shift(1).fillna(0).diff(-1).fillna(0)

    # buy at best ask, sell at best bid (simplied assumption without slippage)
    money_diff = df[bid_column] * np.maximum(strategy_diff, 0)
    money_diff += df[ask_column] * np.minimum(strategy_diff, 0)

    # shift by one. at time zero, integral is always zero
    money_diff = money_diff.shift(1).fillna(0)
    money_account = np.cumsum(money_diff)

    # assumption is that we always cross the spread so pay the fees
    fees = np.cumsum(0.0005 * np.abs(money_diff))

    portfolio_value = (df[strategy_column] * df[bid_column]).shift(1).fillna(0)
    pnl = money_account + portfolio_value

    retval_df = df[[bid_column, ask_column, strategy_column]].rename(
        columns={bid_column: 'bid', ask_column: 'ask', strategy_column: 'strategy'})
    retval_df['fees'] = fees
    retval_df['money'] = money_account
    retval_df['pnl'] = pnl

    return retval_df


def create_strategy(df, gamma=1, com=10):
    # predict returns
    X1, X2 = splitX(df, 1, WINDOW_SIZE)

    # predict returns
    ret_pred = model.predict([X1, X2])
    strategy = np.argmax(ret_pred, axis=1) - 1.

    # calculating ewm strategy
    retval_df = df[['timestamp_diff', 'best_bid_price', 'best_ask_price']].copy()
    retval_df.loc[retval_df.index[WINDOW_SIZE:], ['p-', 'p0', 'p+']] = ret_pred
    retval_df.loc[retval_df.index[WINDOW_SIZE:], 'strategy'] = gamma * strategy.reshape(-1)

    strategy_name = f"ewm_strategy_{com}_{gamma}"
    retval_df.loc[retval_df.index[WINDOW_SIZE:], strategy_name] = np.minimum(1., np.maximum(-1., np.round(
        retval_df['strategy'].ewm(com=com).mean())))
    retval_df = retval_df[~retval_df.isna().any(axis=1)]

    # avoid holding zero on neutral signal, hold -1 or 1 instead continuously until opposite signal prevails
    retval_df[strategy_name] = retval_df[strategy_name].replace(to_replace=0, method='ffill')

    retval_df.loc[:, ['money', 'pnl', 'fees']] = integrate(retval_df, strategy_name)
    retval_df['n_trades'] = np.cumsum(retval_df[strategy_name].diff(1).fillna(0) != 0)

    return retval_df
