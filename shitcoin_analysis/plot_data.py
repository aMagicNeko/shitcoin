import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from util import fetch_pool_keys, WSOL, swap_token_amount_base_in
# 定义处理交易数据的函数
def process_transaction_data(df):
    data = df.copy()

    # Convert timestamps
    data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
    data = data.dropna(subset=['time'])  # 移除无效时间戳
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

    # Filter out sandwich orders
    filtered_data = data.groupby(['time', 'From']).filter(lambda group: not is_sandwich(group))
    
    # Calculate VWAP prices
    buy_orders = filtered_data[filtered_data['type'] == 'buy'].copy()
    sell_orders = filtered_data[filtered_data['type'] == 'sell'].copy()

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
    filtered_data['pool_price'] = filtered_data['Token0'] / filtered_data['Token1']

    # Calculate total inflow and outflow of Token0
    inflow = buy_orders.groupby('time')['Delta0'].sum().reset_index()
    inflow.columns = ['time', 'total_inflow']

    outflow = sell_orders.groupby('time')['Delta0'].sum().reset_index()
    outflow.columns = ['time', 'total_outflow']

    # Get unique time points for pool prices
    pool_data = filtered_data.drop_duplicates(subset=['time']).sort_values(by='time')

    # Merge VWAP, pool price, inflow, and outflow data
    merged_data = pd.merge(order_vwap[['time', 'order_vwap']], pool_data[['time', 'pool_price']], on='time', how='outer')
    merged_data = pd.merge(merged_data, inflow, on='time', how='outer')
    merged_data = pd.merge(merged_data, outflow, on='time', how='outer')

    # Ensure unique time points
    merged_data = merged_data.drop_duplicates(subset=['time'])
    merged_data = merged_data.fillna(0)

    return merged_data, filtered_data

# 定义绘图函数
def plot_inflow_outflow(data, filtered_data, title, strategy_orders):
    # Group by time to ensure unique timestamps and sum the values
    aggregated_data = data.groupby('time').agg({
        'total_inflow': 'sum',
        'total_outflow': 'sum'
    }).reset_index()

    # 计算Token0的累积量
    aggregated_data['cumulative_token0'] = aggregated_data['total_inflow'].cumsum() + aggregated_data['total_outflow'].cumsum()

    plt.figure(figsize=(30, 7))

    # Plot total inflow as scatter plot
    plt.scatter(aggregated_data['time'], aggregated_data['total_inflow'], label='Total Inflow', color='green')

    # Plot total outflow as scatter plot
    plt.scatter(aggregated_data['time'], aggregated_data['total_outflow'], label='Total Outflow', color='red')

    # Plot cumulative Token0 amount as a line plot
    plt.plot(aggregated_data['time'], aggregated_data['cumulative_token0'], label='Cumulative Token0', color='blue')

    # Plot deposits and withdraws as scatter plots
    deposits = filtered_data[filtered_data['type'] == 'Deposit']
    withdraws = filtered_data[filtered_data['type'] == 'Withdraw']
    
    plt.scatter(deposits['time'], deposits['Delta0'], label='Deposits', color='purple', marker='^')
    plt.scatter(withdraws['time'], withdraws['Delta0'], label='Withdraws', color='orange', marker='v')

    # Plot strategy buy and sell orders
    buy_orders = [order for order in strategy_orders if order[1] < 0]
    sell_orders = [order for order in strategy_orders if order[1] > 0]
    
    plt.scatter([order[0] for order in buy_orders], [order[1] for order in buy_orders], label='Strategy Buys', color='yellow', marker='o')
    plt.scatter([order[0] for order in sell_orders], [order[1] for order in sell_orders], label='Strategy Sells', color='black', marker='x')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('Amount of Token0')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Show plot
    plt.show()

def backtest(strategy, df):
    data = df.copy()
    strategy.init_process(data['Token0'][0], data['Token1'][0])
    prev_time = data['time'][0]
    for t_row in data.iterrows():
        row = t_row[1]
        if row['time'] != prev_time:
            strategy.on_tick(row['time'])
        if strategy.end:
            break
        cur_token0 = row['Token0'] + row['Delta0']
        cur_token1 = row['Token1'] + row['Delta1']
        strategy.on_price(cur_token0, cur_token1)
    strategy.on_tick(row['time'])
    out_sol = strategy.nsol + swap_token_amount_base_in(strategy.ntoken, cur_token0, cur_token1, True)
    return out_sol, strategy.my_orders

folder_path = '/Users/ekko/Downloads/coin_data'
def read(file):
    print(f"reading {file}")
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    pool_key = fetch_pool_keys(file.split('_')[0])
    #print(pool_key)
    data = df.copy()
    if pool_key['quote_mint'] == WSOL:
        #print("here")
        data['Token0'], data['Token1'] = df['Token1'], df['Token0']
        data['Delta0'], data['Delta1'] = df['Delta1'], df['Delta0']
    return data