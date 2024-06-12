import pandas as pd
def process_transaction_data(df):
    data = df.copy()

    # Ensure Token0 is always the smaller value for consistency
    if data['Token0'][0] > data['Token1'][0]:
        data['Token0'], data['Token1'] = df['Token1'], df['Token0']
        data['Delta0'], data['Delta1'] = df['Delta1'], df['Delta0']

    # Convert timestamps
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data = data.sort_values('time')

    # Label buy and sell orders
    data['type'] = data.apply(lambda row: 'buy' if row['Delta0'] > 0 and row['Delta1'] < 0 else 'sell' if row['Delta0'] < 0 and row['Delta1'] > 0 else 'Deposit' if row['Delta0'] > 0 and row['Delta1'] > 0 else 'Withdraw', axis=1)

    # Function to detect sandwich orders
    def is_sandwich(group):
        buy_orders = group[group['type'] == 'buy']['Delta0'].sum()
        sell_orders = group[group['type'] == 'sell']['Delta0'].sum()
        if sell_orders == 0:
            return False
        ratio = buy_orders / abs(sell_orders)
        return 0.9 <= ratio <= 1.1 and buy_orders > 1e10

    # Identify sandwich orders
    data['is_sandwich'] = data.groupby(['time', 'From']).transform(lambda group: is_sandwich(group))

    # Separate sandwich and non-sandwich orders
    sandwich_orders = data[data['is_sandwich']]
    non_sandwich_orders = data[~data['is_sandwich']]

    # Calculate VWAP prices for non-sandwich orders
    buy_orders = non_sandwich_orders[non_sandwich_orders['type'] == 'buy'].copy()
    sell_orders = non_sandwich_orders[non_sandwich_orders['type'] == 'sell'].copy()

    buy_orders['vwap'] = buy_orders['Delta0'] / abs(buy_orders['Delta1'])
    sell_orders['vwap'] = abs(sell_orders['Delta0']) / abs(sell_orders['Delta1'])

    buy_vwap = buy_orders.groupby('time').apply(lambda x: (x['vwap'] * abs(x['Delta0'])).sum() / abs(x['Delta0']).sum()).reset_index()
    buy_vwap.columns = ['time', 'buy_vwap']

    sell_vwap = sell_orders.groupby('time').apply(lambda x: (x['vwap'] * abs(x['Delta0'])).sum() / abs(x['Delta0']).sum()).reset_index()
    sell_vwap.columns = ['time', 'sell_vwap']

    # Merge buy and sell VWAP prices
    order_vwap = pd.merge(buy_vwap, sell_vwap, on='time', how='outer')
    order_vwap['order_vwap'] = (order_vwap['buy_vwap'].fillna(0) + order_vwap['sell_vwap'].fillna(0)) / 2

    # Calculate pool price
    non_sandwich_orders['pool_price'] = non_sandwich_orders['Token0'] / non_sandwich_orders['Token1']

    # Calculate total inflow and outflow of Token0
    inflow = buy_orders.groupby('time')['Delta0'].sum().reset_index()
    inflow.columns = ['time', 'total_inflow']

    outflow = sell_orders.groupby('time')['Delta0'].sum().reset_index()
    outflow.columns = ['time', 'total_outflow']

    # Get unique time points for pool prices
    pool_data = non_sandwich_orders.drop_duplicates(subset=['time']).sort_values(by='time')

    # Merge VWAP, pool price, inflow, and outflow data
    merged_data = pd.merge(order_vwap[['time', 'order_vwap']], pool_data[['time', 'pool_price']], on='time', how='outer')
    merged_data = pd.merge(merged_data, inflow, on='time', how='outer')
    merged_data = pd.merge(merged_data, outflow, on='time', how='outer')

    # Ensure unique time points
    merged_data = merged_data.drop_duplicates(subset=['time'])
    merged_data = merged_data.fillna(0)

    return merged_data, non_sandwich_orders, sandwich_orders
