import numpy as np
import pandas as pd


from utils import split_x, get_keys


def integrate(df, strategy, depth=10):

    strategy_diff = -strategy.diff(1).fillna(0)

    execution_price = calculate_execution_price(df.loc[strategy_diff.index], strategy_diff, depth=depth)

    # buy / sell at the execution price
    money_diff = execution_price * np.maximum(strategy_diff, 0)
    money_diff += execution_price * np.minimum(strategy_diff, 0)

    # shift by one. at time zero, integral is always zero
    money_account = np.cumsum(money_diff)

    # assumption is that we always cross the spread so pay the fees
    fees = np.cumsum(0.0005 * np.abs(money_diff))

    mid_price = (df['bp_0'] + df['ap_0']) / 2
    portfolio_value = strategy * mid_price.loc[strategy.index]
    pnl = money_account + portfolio_value

    retval_df = pd.DataFrame(np.array([fees, money_account, pnl]).T, columns=['fees', 'money', 'pnl'], index=strategy.index)

    return retval_df


def calculate_execution_price(df, strategy_diff, depth=10):

    keys, bpcs, apcs, bvcs, avcs = get_keys(depth)

    # calculate price taking into account slippage
    execution_price = pd.Series(0, index=df.index, dtype=np.float64)

    # execution price is calculate from either bids or asks based on the sign of the position
    for factor, mask, pcs, vcs in [(-1, strategy_diff < 0, apcs, avcs), (+1, strategy_diff > 0, bpcs, bvcs)]:

        weights = np.minimum(df.loc[mask, vcs].cumsum(axis=1), factor * strategy_diff.loc[mask][:, np.newaxis])

        # ensure that lob has enough depth
        if not np.all(weights[weights.columns[-1]] == factor * strategy_diff.loc[mask]):
            idx = np.argmin(weights[weights.columns[-1]] == factor * strategy_diff.loc[mask])
            msg = f"Not enough depth in the order buy to buy {strategy_diff[idx]} shares at timestep {weights.index[idx]}"
            raise ValueError(msg)

        # if depth is sufficient, calculate average execution price
        weights = weights.shift(1, axis=1).fillna(0).diff(1, axis=1).shift(-1, axis=1).fillna(0)
        weights = weights.div(weights.sum(axis=1), axis=0)
        execution_price.loc[mask] += np.einsum('ij,ij->i', df.loc[mask, pcs], weights)

    return execution_price


def create_strategy(df, model, position_size=1., depth=10, window_size=100, gamma=20, com=10):

    # predict returns
    X1, X2 = split_x(df, offset=1, window_size=window_size, depth=depth)

    # predict returns
    ret_pred = model.predict([X1, X2])
    strategy = np.argmax(ret_pred, axis=1) - 1.

    # calculating ewm strategy
    retval_df = pd.DataFrame(index=df.index)
    retval_df.loc[retval_df.index[window_size:], ['p-', 'p0', 'p+']] = ret_pred
    retval_df.loc[retval_df.index[window_size:], 'signal'] = gamma * strategy.reshape(-1)

    strategy_name = f"ewm_strategy_{com}_{gamma}"
    retval_df.loc[retval_df.index[window_size:], strategy_name] = np.minimum(1., np.maximum(-1., np.round(retval_df['signal'].ewm(com=com).mean())))
    retval_df = retval_df[~retval_df.isna().any(axis=1)]

    # avoid holding zero on neutral signal, hold -1 or 1 instead continuously until opposite signal prevails
    retval_df[strategy_name] = retval_df[strategy_name].replace(to_replace=0, method='ffill')
    retval_df[strategy_name].iloc[0] = 0
    retval_df[strategy_name] *= position_size

    retval_df.loc[:, ['money', 'pnl', 'fees']] = integrate(df, retval_df[strategy_name])
    retval_df['n_trades'] = np.cumsum(retval_df[strategy_name].diff(1).fillna(0) != 0)

    return retval_df
