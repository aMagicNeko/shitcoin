import os
import polars as pl
import numpy as np
from util import fetch_pool_keys, SOL
from collections import Counter
from scipy.stats import entropy
import logging
from multiprocessing import Pool

def process_transaction_data(df, file_creation_time):
    # Convert slot to time
    df = df.with_column(
        (pl.col('slot') / 3).cast(pl.Duration).alias('time')
    )
    df = df.with_column(
        (file_creation_time + pl.col('time')).alias('datetime')
    )

    # 每秒的开始取Token0的数量
    df = df.with_column(
        pl.col('datetime').dt.truncate('1s').alias('second')
    )
    second_start_data = df.groupby('second').first()

    # 计算各个地址的持仓
    df = df.with_column(
        pl.col('Delta0').cumsum().over('From').alias('cumulative_delta0')
    )

    # 计算每秒每个地址的Delta0和
    data_per_second = df.groupby(['second', 'From']).agg(
        [pl.col('Delta0').sum()]
    )

    return second_start_data, df, data_per_second

def get_time_of_day_feature(time):
    hour = time.hour()
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

def collect_features(data, current_time, cumulative_data, data_per_second):
    features = {}

    # 开盘时间和开盘Token0值
    features['open_time'] = data['datetime'][0]
    features['open_token0'] = data['Token0'][0]
    features['open_token1'] = data['Token1'][0]
    
    # 开盘的流动性
    open_liquidity = data['Token0'][0] * data['Token1'][0]
    features['open_liquidity'] = open_liquidity

    # 当前时间的Token0值
    features['current_time'] = current_time
    current_index = pl.select(pl.col('datetime').searchsorted(current_time, side='right') - 1)
    if current_index >= 0:
        current_token0 = data['Token0'][current_index] + data['Delta0'][current_index]
        current_token1 = data['Token1'][current_index] + data['Delta1'][current_index]
    else:
        current_token0 = data['Token0'][0]  # 默认值为初始Token0值
        current_token1 = data['Token1'][0]  # 默认值为初始Token1值
    features['current_token0'] = current_token0
    features['current_token1'] = current_token1
    
    # 当前流动性与开盘流动性的比值
    current_liquidity = current_token0 * current_token1
    features['current_liquidity_ratio'] = current_liquidity / open_liquidity if open_liquidity != 0 else 0
    
    # 当前Token0与开盘Token0的比值
    features['current_to_open_token0_ratio'] = current_token0 / data['Token0'][0] if data['Token0'][0] != 0 else 0

    # 添加时间段特征
    features['time_of_day'] = get_time_of_day_feature(current_time)

    # 计算不同时间窗口内的Token0值
    time_windows = [15, 30, 90, 180, 360, 900, 1800, 3600, 5400, 10800]  # 3倍时间窗口
    cumulative_data_set = cumulative_data.set_index('datetime')
    data_per_second_set = data_per_second.set_index('second')
    for window in time_windows:
        window_start_time = current_time - pl.duration(seconds=window)
        window_index = pl.select(pl.col('datetime').searchsorted(window_start_time, side='right') - 1)
        if window_index >= 0:
            window_token0 = data['Token0'][window_index] + data['Delta0'][window_index]
        else:
            window_token0 = current_token0  # 使用当前的Token0值作为默认值
        
        features[f'token0_value_{window}s'] = window_token0
        features[f'token0_relative_value_{window}s'] = window_token0 / data['Token0'][0] if data['Token0'][0] != 0 else 0

        if window == time_windows[0]:
            features[f'token0_diff_value_{window}s'] = 0
            features[f'token0_relative_diff_value_{window}s'] = 0
        else:
            prev_window = time_windows[time_windows.index(window) - 1]
            features[f'token0_diff_value_{window}s'] = features[f'token0_value_{window}s'] - features[f'token0_value_{prev_window}s']
            features[f'token0_relative_diff_value_{window}s'] = features[f'token0_relative_value_{window}s'] - features[f'token0_relative_value_{prev_window}s']
    
    # 计算不同时间段的流入和流出Token0
    for window in time_windows:
        window_start_time = current_time - pl.duration(seconds=window)
        window_data = data_per_second_set.filter(pl.col('second') >= window_start_time)
        inflow = window_data.filter(pl.col('Delta0') > 0)['Delta0'].sum()
        outflow = window_data.filter(pl.col('Delta0') < 0)['Delta0'].sum()
        features[f'inflow_{window}s'] = inflow
        features[f'outflow_{window}s'] = outflow

    # 持仓分布特征
    holding_data = cumulative_data_set.filter(pl.col('datetime') <= current_time)
    address_counts = holding_data.groupby('From').agg([pl.col('cumulative_delta0').last()])
    total_token0 = address_counts.filter(pl.col('cumulative_delta0') > 0)['cumulative_delta0'].sum()
    address_proportions = address_counts.filter(pl.col('cumulative_delta0') > 0).with_column(
        (pl.col('cumulative_delta0') / total_token0).alias('proportion')
    )
    
    # 排除持仓为负数的地址
    negative_holdings = address_counts.filter(pl.col('cumulative_delta0') < 0)['cumulative_delta0'].sum()
    features['negative_holdings'] = negative_holdings

    # 地址数量
    features['num_addresses'] = address_proportions.height
    # 最大地址持仓比例
    features['max_address_holding'] = address_proportions['proportion'].max() if not address_proportions.is_empty() else 0
    # 前5大地址持仓比例
    features['top_5_address_holding'] = address_proportions.sort('proportion', reverse=True)[:5]['proportion'].sum() if not address_proportions.is_empty() else 0
    # 前10大地址持仓比例
    features['top_10_address_holding'] = address_proportions.sort('proportion', reverse=True)[:10]['proportion'].sum() if not address_proportions.is_empty() else 0
    # 持仓分布的熵
    features['holding_entropy'] = entropy(address_proportions['proportion']) if not address_proportions.is_empty() else 0

    # 计算不同时间窗口内持仓的差分
    for window in time_windows:
        window_start_time = current_time - pl.duration(seconds=window)
        window_holding_data = cumulative_data_set.filter(pl.col('datetime') <= window_start_time)
        window_address_counts = window_holding_data.groupby('From').agg([pl.col('cumulative_delta0').last()])
        window_total_token0 = window_address_counts.filter(pl.col('cumulative_delta0') > 0)['cumulative_delta0'].sum()
        window_address_proportions = window_address_counts.filter(pl.col('cumulative_delta0') > 0).with_column(
            (pl.col('cumulative_delta0') / window_total_token0).alias('proportion')
        )
        
        if not window_address_proportions.is_empty():
            features[f'top_5_address_holding_diff_{window}s'] = features['top_5_address_holding'] - window_address_proportions.sort('proportion', reverse=True)[:5]['proportion'].sum()
            features[f'top_10_address_holding_diff_{window}s'] = features['top_10_address_holding'] - window_address_proportions.sort('proportion', reverse=True)[:10]['proportion'].sum()
            features[f'max_address_holding_diff_{window}s'] = features['max_address_holding'] - window_address_proportions['proportion'].max()
        else:
            features[f'top_5_address_holding_diff_{window}s'] = 0
            features[f'top_10_address_holding_diff_{window}s'] = 0
            features[f'max_address_holding_diff_{window}s'] = 0

    return features

def collect_targets(data, current_time):
    targets = {}

    # 获取当前时间的Token0值
    current_index = pl.select(pl.col('datetime').searchsorted(current_time, side='right') - 1)
    if current_index >= 0:
        current_token0 = data['Token0'][current_index] + data['Delta0'][current_index]
    else:
        current_token0 = data['Token0'][0]  # 默认值为初始Token0值

    # 计算不同时间窗口内的Token0值
    time_windows = [15, 30, 90, 180, 900, 1800, 5400, 10800]  # 3倍时间窗口
    for window in time_windows:
        window_end_time = current_time + pl.duration(seconds=window)
        window_index = pl.select(pl.col('datetime').searchsorted(window_end_time, side='right') - 1)
        if window_index >= 0:
            window_token0 = data['Token0'][window_index] + data['Delta0'][window_index]
        else:
            window_token0 = current_token0  # 使用当前的Token0值作为默认值

        targets[f'token0_value_{window}s'] = window_token0

    # Add target for dramatic drop in Token0
    drop_thresholds = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    for window in [6, 15, 30]:  # 调整为3倍时间窗口
        for threshold in drop_thresholds:
            window_end_time = current_time + pl.duration(seconds=window)
            window_index = pl.select(pl.col('datetime').searchsorted(window_end_time, side='right') - 1)
            if window_index >= 0:
                window_token0 = data['Token0'][window_index] + data['Delta0'][window_index]
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
            if (data['Token0'][i] + data['Delta0'][i] > current_token0 * (1 + threshold) and
                data['Token0'][i + 1] + data['Delta0'][i + 1] > current_token0 * (1 + threshold)):
                consecutive_rise = True
                break
        targets[f'token0_consecutive_rise_{threshold*100:.0f}%'] = 1 if consecutive_rise else 0

    return targets

def read_and_process(file_path):
    print(f"start to process {file_path}")
    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None, None

    if df.is_empty() or 'slot' not in df.columns:
        print(f"File {file_path} is empty or does not contain 'slot' field.")
        return None, None

    file_creation_time = pl.datetime(os.path.getctime(file_path), unit='s')
    
    pool_address = os.path.basename(file_path).split('_')[0]
    if fetch_pool_keys(pool_address)['base_mint'] != SOL:
        df = df.with_columns([
            pl.col('Token0').alias('Token1'),
            pl.col('Token1').alias('Token0'),
            pl.col('Delta0').alias('Delta1'),
            pl.col('Delta1').alias('Delta0')
        ])
    
    try:
        processed_data, cumulative_data, data_per_second = process_transaction_data(df, file_creation_time)
        all_features = []
        all_targets = []
        for _, row in processed_data.iter_rows(named=True):
            current_time = row['datetime']
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
        features_df = pl.DataFrame(features)
        targets_df = pl.DataFrame(targets)
        output_dir = "processed_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        features_df.write_csv(os.path.join(output_dir, f"{os.path.basename(file_path)}_features.csv"))
        targets_df.write_csv(os.path.join(output_dir, f"{os.path.basename(file_path)}_targets.csv"))
    else:
        print(f"Failed to process {file_path}")

if __name__ == "__main__":
    import glob

    # 获取所有需要处理的文件路径
    file_paths = glob.glob("../coin_data/*.parquet")

    # 打印获取到的文件路径
    print(f"Found {len(file_paths)} files to process.")

    # 使用多进程处理文件
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_file, file_paths)
