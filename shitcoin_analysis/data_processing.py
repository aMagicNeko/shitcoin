# data_processing.py
import os
import pandas as pd
import numpy as np
from util import fetch_pool_keys, SOL

# 定义处理交易数据的函数
def process_transaction_data(df):
    data = df.copy()

    # Convert timestamps
    data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
    data = data.dropna(subset=['time'])  # 移除无效时间戳
    data = data.sort_values('time')

    # 每秒的末尾取Token0的数量
    data['second'] = data['time'].dt.floor('S')
    second_end_data = data.groupby('second').last().reset_index()

    return second_end_data

# 收集特征
def collect_features(data, current_time):
    features = {}
    
    # 开盘时间和开盘Token0值
    features['open_time'] = data.iloc[0]['time']
    features['open_token0'] = data.iloc[0]['Token0']
    
    # 当前时间的Token0值
    features['current_time'] = current_time
    current_token0 = data.loc[data['time'] == current_time, 'Token0'].values[0]
    features['current_token0'] = current_token0

    # 计算不同时间窗口内的平均Token0值
    time_windows = [5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
    for window in time_windows:
        window_data = data[(data['time'] >= current_time - pd.Timedelta(seconds=window)) & (data['time'] < current_time)]
        features[f'avg_token0_{window}s'] = window_data['Token0'].mean() if not window_data.empty else np.nan

    return features

# 收集目标变量
def collect_targets(data, current_time):
    targets = {}
    
    # 计算不同时间窗口内的最大涨幅和最大跌幅
    time_windows = [10, 30, 60, 300, 600, 1800, 3600, len(data)]
    current_token0 = data.loc[data['time'] == current_time, 'Token0'].values[0]
    for window in time_windows:
        future_data = data[(data['time'] > current_time) & (data['time'] <= current_time + pd.Timedelta(seconds=window))]
        if not future_data.empty:
            max_price = future_data['Token0'].max()
            min_price = future_data['Token0'].min()
            targets[f'max_increase_{window}s'] = (max_price - current_token0) / current_token0
            targets[f'max_decrease_{window}s'] = (min_price - current_token0) / current_token0
        else:
            targets[f'max_increase_{window}s'] = np.nan
            targets[f'max_decrease_{window}s'] = np.nan

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
        processed_data = process_transaction_data(data)
        all_features = []
        all_targets = []
        for _, row in processed_data.iterrows():
            current_time = row['time']
            features = collect_features(processed_data, current_time)
            targets = collect_targets(processed_data, current_time)
            
            all_features.append(features)
            all_targets.append(targets)
        return all_features, all_targets
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None
