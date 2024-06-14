import os
import pandas as pd
import numpy as np
from util import fetch_pool_keys, SOL
from collections import Counter
from scipy.stats import entropy
import logging
from multiprocessing import Pool

# 定义处理交易数据的函数
def process_transaction_data(df):
    # Convert timestamps
    df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')

    # 每秒的开始取Token0的数量
    df['second'] = df['time'].dt.floor('S')
    second_start_data = df.groupby('second').first().reset_index()

    # 计算各个地址的持仓
    df['cumulative_delta0'] = df.groupby('From')['Delta0'].cumsum()

    # 计算每秒每个地址的Delta0和
    data_per_second = df.groupby(['second', 'From']).agg({'Delta0': 'sum'}).reset_index()

    return second_start_data, df, data_per_second

def collect_features(data, current_time, cumulative_data, data_per_second):
    features = {}

    # 开盘时间和开盘Token0值
    features['open_time'] = data.iloc[0]['time']
    features['open_token0'] = data.iloc[0]['Token0']
    features['open_token1'] = data.iloc[0]['Token1']
    
    # 开盘的流动性
    open_liquidity = data.iloc[0]['Token0'] * data.iloc[0]['Token1']
    features['open_liquidity'] = open_liquidity

    # 当前时间的Token0值
    features['current_time'] = current_time
    current_index = data['time'].searchsorted(current_time, side='right') - 1
    if current_index >= 0:
        current_token0 = data.iloc[current_index]['Token0'] + data.iloc[current_index]['Delta0']
        current_token1 = data.iloc[current_index]['Token1'] + data.iloc[current_index]['Delta1']
    else:
        current_token0 = data.iloc[0]['Token0']  # 默认值为初始Token0值
        current_token1 = data.iloc[0]['Token1']  # 默认值为初始Token1值
    features['current_token0'] = current_token0
    features['current_token1'] = current_token1
    
    # 当前流动性与开盘流动性的比值
    current_liquidity = current_token0 * current_token1
    features['current_liquidity_ratio'] = current_liquidity / open_liquidity if open_liquidity != 0 else 0
    
    # 当前Token0与开盘Token0的比值
    features['current_to_open_token0_ratio'] = current_token0 / data.iloc[0]['Token0'] if data.iloc[0]['Token0'] != 0 else 0

    # 计算不同时间窗口内的Token0值
    time_windows = [5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
    cumulative_data_set = cumulative_data.set_index('time')
    data_per_second_set = data_per_second.set_index('second')
    for window in time_windows:
        window_start_time = current_time - pd.Timedelta(seconds=window)
        window_index = data['time'].searchsorted(window_start_time, side='right') - 1
        if window_index >= 0:
            # 使用每一行的 `Token0` 值加上 `Delta0` 值
            window_token0 = data.iloc[window_index]['Token0'] + data.iloc[window_index]['Delta0']
        else:
            window_token0 = current_token0  # 使用当前的Token0值作为默认值
        
        # Token0流动量的相对流动量
        features[f'token0_value_{window}s'] = window_token0
        features[f'token0_relative_value_{window}s'] = window_token0 / data.iloc[0]['Token0'] if data.iloc[0]['Token0'] != 0 else 0

        # 计算流动量差分值
        if window == time_windows[0]:
            features[f'token0_diff_value_{window}s'] = 0
            features[f'token0_relative_diff_value_{window}s'] = 0
        else:
            prev_window = time_windows[time_windows.index(window) - 1]
            features[f'token0_diff_value_{window}s'] = features[f'token0_value_{window}s'] - features[f'token0_value_{prev_window}s']
            features[f'token0_relative_diff_value_{window}s'] = features[f'token0_relative_value_{window}s'] - features[f'token0_relative_value_{prev_window}s']
    
    # 计算不同时间段的流入和流出Token0
    for window in time_windows:
        window_start_time = current_time - pd.Timedelta(seconds=window)
        window_data = data_per_second_set.loc[window_start_time:current_time]
        inflow = window_data[window_data['Delta0'] > 0]['Delta0'].sum()
        outflow = window_data[window_data['Delta0'] < 0]['Delta0'].sum()
        features[f'inflow_{window}s'] = inflow
        features[f'outflow_{window}s'] = outflow

    # 持仓分布特征
    holding_data = cumulative_data_set.loc[:current_time]
    address_counts = holding_data.groupby('From')['cumulative_delta0'].last()
    total_token0 = address_counts[address_counts > 0].sum()
    address_proportions = address_counts[address_counts > 0] / total_token0
    
    # 排除持仓为负数的地址
    negative_holdings = address_counts[address_counts < 0].sum()
    features['negative_holdings'] = negative_holdings

    # 地址数量
    features['num_addresses'] = address_proportions.size
    # 最大地址持仓比例
    features['max_address_holding'] = address_proportions.max() if not address_proportions.empty else 0
    # 前5大地址持仓比例
    features['top_5_address_holding'] = address_proportions.nlargest(5).sum() if not address_proportions.empty else 0
    # 前10大地址持仓比例
    features['top_10_address_holding'] = address_proportions.nlargest(10).sum() if not address_proportions.empty else 0
    # 持仓分布的熵
    features['holding_entropy'] = entropy(address_proportions) if not address_proportions.empty else 0

    # 计算不同时间窗口内持仓的差分
    for window in time_windows:
        window_start_time = current_time - pd.Timedelta(seconds=window)
        window_holding_data = cumulative_data_set.loc[:window_start_time]
        window_address_counts = window_holding_data.groupby('From')['cumulative_delta0'].last()
        window_total_token0 = window_address_counts[window_address_counts > 0].sum()
        window_address_proportions = window_address_counts[window_address_counts > 0] / window_total_token0
        
        if not window_address_proportions.empty:
            # 差分计算
            features[f'top_5_address_holding_diff_{window}s'] = features['top_5_address_holding'] - window_address_proportions.nlargest(5).sum()
            features[f'top_10_address_holding_diff_{window}s'] = features['top_10_address_holding'] - window_address_proportions.nlargest(10).sum()
            features[f'max_address_holding_diff_{window}s'] = features['max_address_holding'] - window_address_proportions.max()
        else:
            features[f'top_5_address_holding_diff_{window}s'] = 0
            features[f'top_10_address_holding_diff_{window}s'] = 0
            features[f'max_address_holding_diff_{window}s'] = 0

    return features

def collect_targets(data, current_time):
    targets = {}

    # 获取当前时间的Token0值
    current_index = data['time'].searchsorted(current_time, side='right') - 1
    if current_index >= 0:
        current_token0 = data.iloc[current_index]['Token0'] + data.iloc[current_index]['Delta0']
    else:
        current_token0 = data.iloc[0]['Token0']  # 默认值为初始Token0值

    # 计算不同时间窗口内的Token0值
    time_windows = [5, 10, 30, 60, 300, 600, 1800, 3600]
    for window in time_windows:
        window_end_time = current_time + pd.Timedelta(seconds=window)
        window_index = data['time'].searchsorted(window_end_time, side='right') - 1
        if window_index >= 0:
            window_token0 = data.iloc[window_index]['Token0'] + data.iloc[window_index]['Delta0']
        else:
            window_token0 = current_token0  # 使用当前的Token0值作为默认值

        targets[f'token0_value_{window}s'] = window_token0

    # Add target for dramatic drop in Token0
    drop_thresholds = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    for window in [2, 5, 10]:
        for threshold in drop_thresholds:
            window_end_time = current_time + pd.Timedelta(seconds=window)
            window_index = data['time'].searchsorted(window_end_time, side='right') - 1
            if window_index >= 0:
                window_token0 = data.iloc[window_index]['Token0'] + data.iloc[window_index]['Delta0']
            else:
                window_token0 = current_token0  # 使用当前的Token0值作为默认值

            if window_token0 < current_token0 * (1 - threshold):
                targets[f'token0_drop_{threshold*100:.0f}%_{window}s'] = 1
            else:
                targets[f'token0_drop_{threshold*100:.0f}%_{window}s'] = 0

    # Add target for consecutive two-second rise in Token0
    rise_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    for threshold in rise_thresholds:
        consecutive_rise = False
        for i in range(1, len(data) - 1):
            if (data.iloc[i]['Token0'] + data.iloc[i]['Delta0'] > current_token0 * (1 + threshold) and
                data.iloc[i + 1]['Token0'] + data.iloc[i + 1]['Delta0'] > current_token0 * (1 + threshold)):
                consecutive_rise = True
                break
        targets[f'token0_consecutive_rise_{threshold*100:.0f}%'] = 1 if consecutive_rise else 0

    return targets


def read_and_process(file_path):
    print(f"start to process {file_path}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None, None

    if df.empty or 'time' not in df.columns:
        print(f"File {file_path} is empty or does not contain 'time' field.")
        return None, None

    pool_address = os.path.basename(file_path).split('_')[0]
    if fetch_pool_keys(pool_address)['base_mint'] != SOL:
        df[['Token0', 'Token1']] = df[['Token1', 'Token0']]
        df[['Delta0', 'Delta1']] = df[['Delta1', 'Delta0']]
    
    try:
        processed_data, cumulative_data, data_per_second = process_transaction_data(df)
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

def process_file(file_path):
    print(f"Processing file: {file_path}")
    features, targets = read_and_process(file_path)
    if features is not None and targets is not None:
        features_df = pd.DataFrame(features)
        targets_df = pd.DataFrame(targets)
        output_dir = "processed_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        features_df.to_csv(os.path.join(output_dir, f"{os.path.basename(file_path)}_features.csv"), index=False)
        targets_df.to_csv(os.path.join(output_dir, f"{os.path.basename(file_path)}_targets.csv"), index=False)
    else:
        print(f"Failed to process {file_path}")

if __name__ == "__main__":
    import glob

    # 获取所有需要处理的文件路径
    file_paths = glob.glob("../coin_data/*.xlsx")

    # 打印获取到的文件路径
    print(f"Found {len(file_paths)} files to process.")

    # 使用多进程处理文件
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_file, file_paths)
