import os
import polars as pl
import numpy as np
from util import fetch_pool_keys, SOL
from collections import Counter
from scipy.stats import entropy
import logging
from multiprocessing import Pool
from datetime import datetime, timedelta

def process_transaction_data(df: pl.DataFrame, file_creation_time):
    # Calculate datetime based on slot and file creation time
    time_diff = df['slot'] * 400 / 1000
    datetime_column = [file_creation_time + timedelta(milliseconds=int(td)) for td in time_diff]

    # Add the datetime column
    df = df.with_columns([
        pl.Series(name='datetime', values=datetime_column)
    ])

    # Create the second column based on the newly created datetime column
    df = df.with_columns([
        pl.col('datetime').dt.truncate('1s').alias('second')
    ])

    # Select the last value of each column for each slot
    columns_to_aggregate = [pl.col(column).last().alias(f"{column}_end") for column in df.columns]
    slot_end_data = df.groupby('slot').agg(columns_to_aggregate)

    # Calculate cumulative Delta0 for each address
    df = df.with_columns([
        pl.col('Delta0').cumsum().over('From').alias('cumulative_delta0')
    ])

    return slot_end_data, df

def get_time_of_day_feature(time):
    hour = time.hour
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

def collect_features(data: pl.DataFrame, row, cumulative_data):
    features = {}
    current_slot = row['slot']
    # Ensure columns are accessible
    data = data.select([col for col in data.columns])

    # 开盘时间和开盘Token0值
    open_time = cumulative_data[0, 'datetime']
    open_token0 = cumulative_data[0, 'Token0']
    open_token1 = cumulative_data[0, 'Token1']
    features['open_time'] = open_time
    features['open_token0'] = open_token0
    features['open_token1'] = open_token1
    
    # 开盘的流动性
    open_liquidity = float(open_token0) * open_token1
    features['open_liquidity'] = open_liquidity

    # 当前slot的Token0值
    features['current_slot'] = current_slot
    #current_index = np.searchsorted(data['slot'].to_numpy(), current_slot, side='right') - 1
    current_token0 = row['Token0_end'] + row['Delta0_end']
    current_token1 = row['Token1_end'] + row['Delta1_end']

    features['current_token0'] = current_token0
    features['current_token1'] = current_token1
    
    # 当前流动性与开盘流动性的比值
    current_liquidity = float(current_token0) * current_token1
    features['current_liquidity_ratio'] = current_liquidity / open_liquidity if open_liquidity != 0 else 0
    
    # 当前Token0与开盘Token0的比值
    features['current_to_open_token0_ratio'] = current_token0 / open_token0 if open_token0 != 0 else 0

    # 添加时间段特征
    current_time = row['datetime_end']
    features['time_of_day'] = get_time_of_day_feature(current_time)

    # 计算不同slot窗口内的Token0值
    slot_windows = [15, 30, 60, 120, 240, 480, 960, 1920, 3840, 7680]  # 3倍时间窗口
    for window in slot_windows:
        window_start_slot = current_slot - window
        window_index = int(np.searchsorted(data['slot'].to_numpy(), window_start_slot, side='right') - 1)
        if window_index >= 0:
            window_token0 = data[window_index, 'Token0_end'] + data[window_index, 'Delta0_end']
        else:
            window_token0 = open_token0
        
        features[f'token0_value_{window}slots'] = window_token0
        features[f'token0_relative_value_{window}slots'] = window_token0 / open_token0 if open_token0 != 0 else 0
        # difference of flow?
        if window == slot_windows[0]:
            features[f'token0_diff_value_{window}slots'] = 0
            features[f'token0_relative_diff_value_{window}slots'] = 0
        else:
            prev_window = slot_windows[slot_windows.index(window) - 1]
            features[f'token0_diff_value_{prev_window}slots'] = features[f'token0_value_{window}slots'] - features[f'token0_value_{prev_window}slots']
            features[f'token0_relative_diff_value_{prev_window}slots'] = features[f'token0_relative_value_{window}slots'] - features[f'token0_relative_value_{prev_window}slots']

    # 计算不同slot窗口内的流入和流出Token0
    for window in slot_windows:
        window_start_slot = current_slot - window
        window_data = cumulative_data.filter(pl.col('slot') >= window_start_slot)
        inflow = window_data.filter(pl.col('Delta0') > 0)['Delta0'].sum()
        outflow = window_data.filter(pl.col('Delta0') < 0)['Delta0'].sum()
        features[f'inflow_{window}slots'] = inflow
        features[f'outflow_{window}slots'] = outflow

    # 持仓分布特征
    holding_data = cumulative_data.filter(pl.col('slot') <= current_slot)
    address_counts = holding_data.groupby('From').agg([pl.col('cumulative_delta0').last()])
    total_token0 = address_counts.filter(pl.col('cumulative_delta0') > 0)['cumulative_delta0'].sum()
    address_proportions = address_counts.filter(pl.col('cumulative_delta0') > 0).with_columns([
        (pl.col('cumulative_delta0') / total_token0).alias('proportion')
    ])
    
    # 排除持仓为负数的地址
    negative_holdings = address_counts.filter(pl.col('cumulative_delta0') < 0)['cumulative_delta0'].sum()
    features['negative_holdings'] = negative_holdings

    # 地址数量
    features['num_addresses'] = address_proportions.height
    # 最大地址持仓比例
    features['max_address_holding'] = address_proportions['proportion'].max() if not address_proportions.is_empty() else 0
    # 前5大地址持仓比例
    features['top_5_address_holding'] = address_proportions.sort('proportion', descending=True)[:5]['proportion'].sum() if not address_proportions.is_empty() else 0
    # 前10大地址持仓比例
    features['top_10_address_holding'] = address_proportions.sort('proportion', descending=True)[:10]['proportion'].sum() if not address_proportions.is_empty() else 0
    # 持仓分布的熵
    features['holding_entropy'] = entropy(address_proportions['proportion']) if not address_proportions.is_empty() else 0

    # 计算不同slot窗口内持仓的差分
    for window in slot_windows:
        window_start_slot = current_slot - window
        window_holding_data = cumulative_data.filter(pl.col('slot') <= window_start_slot)
        window_address_counts = window_holding_data.groupby('From').agg([pl.col('cumulative_delta0').last()])
        window_total_token0 = window_address_counts.filter(pl.col('cumulative_delta0') > 0)['cumulative_delta0'].sum()
        window_address_proportions = window_address_counts.filter(pl.col('cumulative_delta0') > 0).with_columns([
            (pl.col('cumulative_delta0') / window_total_token0).alias('proportion')
        ])
        
        if not window_address_proportions.is_empty():
            features[f'top_5_address_holding_diff_{window}slots'] = features['top_5_address_holding'] - window_address_proportions.sort('proportion', descending=True)[:5]['proportion'].sum()
            features[f'top_10_address_holding_diff_{window}slots'] = features['top_10_address_holding'] - window_address_proportions.sort('proportion', descending=True)[:10]['proportion'].sum()
            features[f'max_address_holding_diff_{window}slots'] = features['max_address_holding'] - window_address_proportions['proportion'].max()
        else:
            features[f'top_5_address_holding_diff_{window}slots'] = 0
            features[f'top_10_address_holding_diff_{window}slots'] = 0
            features[f'max_address_holding_diff_{window}slots'] = 0

    return features

def collect_targets(data, current_slot):
    targets = {}

    # 获取当前slot的Token0值
    current_index = int(np.searchsorted(data['slot'].to_numpy(), current_slot, side='right') - 1)
    if current_index >= 0:
        current_token0 = data[current_index, 'Token0_end'] + data[current_index, 'Delta0_end']
    else:
        current_token0 = data[0, 'Token0_end']  # 默认值为初始Token0值

    # 计算不同slot窗口内的Token0值
    slot_windows = [15, 30, 60, 120, 240, 480, 960, 1920, 3840, 7680]  # 3倍时间窗口
    for window in slot_windows:
        window_end_slot = current_slot + window
        window_index = int(np.searchsorted(data['slot'].to_numpy(), window_end_slot, side='right') - 1)
        if window_index >= 0:
            window_token0 = data[window_index, 'Token0_end'] + data[window_index, 'Delta0_end']
        else:
            window_token0 = current_token0  # 使用当前的Token0值作为默认值

        targets[f'token0_value_{window}slots'] = window_token0

    # Add target for dramatic drop in Token0
    drop_thresholds = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    for window in [6, 15, 30, 60]:  # 调整为3倍slot窗口
        for threshold in drop_thresholds:
            window_end_slot = current_slot + window
            window_index = int(np.searchsorted(data['slot'].to_numpy(), window_end_slot, side='right') - 1)
            if window_index >= 0:
                window_token0 = data[window_index, 'Token0_end'] + data[window_index, 'Delta0_end']
            else:
                window_token0 = current_token0  # 使用当前的Token0值作为默认值

            if window_token0 < current_token0 * (1 - threshold):
                targets[f'token0_drop_{threshold*100:.0f}%_{window}slots'] = 1
            else:
                targets[f'token0_drop_{threshold*100:.0f}%_{window}slots'] = 0

    # Add target for consecutive ten-slot rise in Token0
    rise_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    for threshold in rise_thresholds:
        consecutive_rise = False
        for i in range(0, len(data) - 10):
            if all(data[i + j, 'Token0_end'] + data[i + j, 'Delta0_end'] > current_token0 * (1 + threshold) for j in range(10)):
                consecutive_rise = True
                break
        targets[f'token0_consecutive_rise_{threshold*100:.0f}%'] = 1 if consecutive_rise else 0

    return targets

def read_and_process(file_path, date):
    print(f"start to process {file_path}")
    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None, None

    if df.is_empty() or 'slot' not in df.columns:
        print(f"File {file_path} is empty or does not contain 'slot' field.")
        return None, None

    file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
    
    #try:
    processed_data, cumulative_data = process_transaction_data(df, file_creation_time)
    all_features = []
    all_targets = []
    for row in processed_data.iter_rows(named=True):
        current_slot = row['slot_end']
        features = collect_features(processed_data, row, cumulative_data)
        targets = collect_targets(processed_data, current_slot)
        
        all_features.append(features)
        all_targets.append(targets)
    return all_features, all_targets
    #except Exception as e:
    #print(f"Error processing {file_path}: {e}")
    return None, None

def process_file(file_path):
    print(f"Processing file: {file_path}")
    date = os.path.basename(os.path.dirname(file_path))
    features, targets = read_and_process(file_path, date)
    if features is not None and targets is not None:
        features_df = pl.DataFrame(features)
        targets_df = pl.DataFrame(targets)
        output_dir = os.path.join("processed_data", date)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        features_df.write_parquet(os.path.join(output_dir, f"{os.path.basename(file_path)}_features.parquet"))
        targets_df.write_parquet(os.path.join(output_dir, f"{os.path.basename(file_path)}_targets.parquet"))
    else:
        print(f"Failed to process {file_path}")

def get_all_parquet_files(root_dir):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.parquet'):
                file_paths.append(os.path.join(root, file))
    return file_paths

if __name__ == "__main__":
    process_file("1XXZVgHFf6suGZ4Je5x7Rz67gHFA6PjSMLpEofz4wB4.parquet")
    root_dir = r"..\coin_data"  # Windows路径
    file_paths = get_all_parquet_files(root_dir)

    # 打印获取到的文件路径
    print(f"Found {len(file_paths)} files to process.")
    for file in file_paths:
        process_file(file)
