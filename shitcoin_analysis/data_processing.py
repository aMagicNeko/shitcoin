# data_processing.py
import os
import pandas as pd
import numpy as np
from util import fetch_pool_keys, SOL
from collections import Counter
from scipy.stats import entropy

# 定义处理交易数据的函数
def process_transaction_data(df):
    data = df.copy()

    # Convert timestamps
    data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
    data = data.dropna(subset=['time'])  # 移除无效时间戳
    data = data.sort_values('time')

    # 每秒的开始取Token0的数量
    data['second'] = data['time'].dt.floor('S')
    second_start_data = data.groupby('second').first().reset_index()

    # 计算各个地址的持仓
    data['cumulative_delta0'] = data.groupby('From')['Delta0'].cumsum()

    # 计算每秒每个地址的Delta0和
    data_per_second = data.groupby(['second', 'From']).agg({'Delta0': 'sum'}).reset_index()

    return second_start_data, data, data_per_second

# 收集特征
def collect_features(data, current_time, cumulative_data, data_per_second):
    features = {}
    
    # 开盘时间和开盘Token0值
    features['open_time'] = data.iloc[0]['time']
    features['open_token0'] = data.iloc[0]['Token0']
    
    # 当前时间的Token0值
    features['current_time'] = current_time
    current_token0 = data.loc[data['time'] == current_time, 'Token0'].values[0]
    features['current_token0'] = current_token0

    # 计算不同时间窗口内的Token0值
    time_windows = [5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
    for window in time_windows:
        window_start_time = current_time - pd.Timedelta(seconds=window)
        window_data = data[(data['time'] > window_start_time) & (data['time'] <= current_time)]

        if not window_data.empty:
            token0_value = window_data.iloc[0]['Token0']
        else:
            token0_value = current_token0 - cumulative_data[(cumulative_data['time'] > window_start_time) & (cumulative_data['time'] <= current_time)]['Delta0'].sum()

        features[f'token0_value_{window}s'] = token0_value

    # 计算不同时间段的流入和流出Token0
    for window in time_windows:
        window_data = data_per_second[(data_per_second['second'] > current_time - pd.Timedelta(seconds=window)) & (data_per_second['second'] <= current_time)]
        inflow = window_data[window_data['Delta0'] > 0]['Delta0'].sum()
        outflow = window_data[window_data['Delta0'] < 0]['Delta0'].sum()
        features[f'inflow_{window}s'] = inflow
        features[f'outflow_{window}s'] = outflow

    # 持仓分布特征
    holding_data = cumulative_data[(cumulative_data['time'] <= current_time)]
    address_counts = holding_data.groupby('From')['cumulative_delta0'].last()
    total_token0 = address_counts.sum()
    address_proportions = [count / total_token0 for count in address_counts]

    # 地址数量
    features['num_addresses'] = len(address_counts)
    # 最大地址持仓比例
    features['max_address_holding'] = max(address_proportions)
    # 前5大地址持仓比例
    features['top_5_address_holding'] = sum(sorted(address_proportions, reverse=True)[:5])
    # 前10大地址持仓比例
    features['top_10_address_holding'] = sum(sorted(address_proportions, reverse=True)[:10])
    # 持仓分布的熵
    features['holding_entropy'] = entropy(address_proportions)

    return features

# 填补时间序列数据的函数
def fill_time_series(data, start_time, end_time):
    time_range = pd.date_range(start=start_time, end=end_time, freq='S')
    data = data.set_index('time').reindex(time_range).ffill().reset_index()
    data = data.rename(columns={'index': 'time'})
    return data

# 收集目标变量
def collect_targets(data, current_time):
    targets = {}
    
    # 填补时间序列
    start_time = data['time'].min()
    end_time = data['time'].max()
    filled_data = fill_time_series(data, start_time, end_time)
    
    # 计算不同时间窗口内的最大涨幅和最大跌幅
    time_windows = [10, 30, 60, 300, 600, 1800, 3600]
    current_token0 = filled_data.loc[filled_data['time'] == current_time, 'Token0'].values[0]
    for window in time_windows:
        future_data = filled_data[(filled_data['time'] > current_time) & (filled_data['time'] <= current_time + pd.Timedelta(seconds=window))]
        if not future_data.empty:
            max_price = future_data['Token0'].max()
            min_price = future_data['Token0'].min()
            targets[f'max_increase_{window}s'] = max_price
            targets[f'max_decrease_{window}s'] = min_price
        else:
            targets[f'max_increase_{window}s'] = current_token0  # 没有数据变化，保持当前值
            targets[f'max_decrease_{window}s'] = current_token0  # 没有数据变化，保持当前值

    # 计算整个时间段内的最大涨幅和最大跌幅
    future_data = filled_data[filled_data['time'] > current_time]
    if not future_data.empty:
        max_price = future_data['Token0'].max()
        min_price = future_data['Token0'].min()
        targets['max_increase_all'] = max_price
        targets['max_decrease_all'] = min_price
    else:
        targets['max_increase_all'] = current_token0  # 没有数据变化，保持当前值
        targets['max_decrease_all'] = current_token0  # 没有数据变化，保持当前值

    return targets

def read_and_process(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None, None

    if df.empty or 'time' not in df.columns:
        print(f"File {file_path} is empty or does not contain 'time' field.")
        return None, None

    data = df.copy()
    pool_address = os.path.basename(file_path).split('_')[0]
    if fetch_pool_keys(pool_address)['base_mint'] != SOL:
        data['Token0'], data['Token1'] = df['Token1'], df['Token0']
        data['Delta0'], data['Delta1'] = df['Delta1'], df['Delta0']
    
    try:
        processed_data, cumulative_data, data_per_second = process_transaction_data(data)
        all_features = []
        all_targets = []
        for _, row in processed_data.iterrows():
            current_time = row['time']
            features = collect_features(processed_data, current_time, cumulative_data, data_per_second)
            targets = collect_targets(processed_data, current_time)
            
            all_features.append(features)
            all_targets.append(targets)
        return all_features, all_targets
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None
