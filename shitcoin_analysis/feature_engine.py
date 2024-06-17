import os
import polars as pl
import numpy as np
from collections import defaultdict
from scipy.stats import entropy
from datetime import datetime, timedelta

import os
import polars as pl
import numpy as np
from collections import defaultdict
from scipy.stats import entropy
from datetime import datetime, timedelta
def process_transaction_data(df: pl.DataFrame, file_creation_time):
    # Calculate datetime based on slot and file creation time
    time_diff = (df['slot'] - df['slot'][0]) * 400
    datetime_column = [file_creation_time + timedelta(milliseconds=int(td)) for td in time_diff]
    # Add the datetime column
    df = df.with_columns([
        pl.Series(name='datetime', values=datetime_column)
    ])

    # Calculate net inflow and outflow within each slot for each address
    net_delta = df.groupby(['slot', 'From']).agg([
        pl.col('Delta0').sum().alias('net_delta0')
    ])

    # Join net delta back to the original DataFrame

    # Select the last value of each column for each slot, excluding 'From' and 'Delta0'/'Delta1'
    columns_to_aggregate = [
        (pl.col('Token0') + pl.col('Delta0')).last().alias('Token0_end'),
        (pl.col('Token1') + pl.col('Delta1')).last().alias('Token1_end')
    ] + [pl.col(column).last().alias(f"{column}_end") for column in df.columns if column not in ['From', 'Delta0', 'Delta1', 'Token0', 'Token1', 'slot']]

    slot_end_data = df.groupby('slot').agg(columns_to_aggregate)
    df = df.join(net_delta, on=['slot', 'From'])
    df = df.sort('slot') # might change order after join
    # Calculate cumulative inflow and outflow
    df = df.with_columns([
        (pl.when(pl.col('net_delta0') > 0).then(pl.col('net_delta0')).otherwise(0).cumsum()).alias('cumulative_inflow'),
        (pl.when(pl.col('net_delta0') < 0).then(pl.col('net_delta0')).otherwise(0).cumsum()).alias('cumulative_outflow')
    ])

    # Aggregate cumulative inflow and outflow for each slot
    inflow_outflow_data = df.groupby('slot').agg([
        pl.col('cumulative_inflow').last().alias('cumulative_inflow'),
        pl.col('cumulative_outflow').last().alias('cumulative_outflow')
    ])

    # Join cumulative inflow and outflow to slot_end_data
    slot_end_data = slot_end_data.join(inflow_outflow_data, on='slot')
    slot_end_data = slot_end_data.sort('slot')
    slot_end_data = slot_end_data.with_columns([pl.col("Token0_end").alias("Token0"), pl.col("Token1_end").alias("Token1"), pl.col("datetime_end").alias("datetime")])
    slot_end_data = slot_end_data.drop('Token0_end')
    slot_end_data = slot_end_data.drop('Token1_end')
    slot_end_data = slot_end_data.drop("datetime_end")

    return slot_end_data, df.groupby(['slot', 'From']).agg(pl.col('net_delta0').sum().alias('cumulative_delta0')).sort('slot')

def get_time_of_day_feature(datetime_column: pl.DataFrame):
    return (
        datetime_column.dt.hour()
        .apply(lambda hour: 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening' if 18 <= hour < 24 else 'night')
    ).alias('time_of_day')

def fill_missing_slots(slot_end_data, all_slots):
    # Join to fill in missing slots and sort by slot to ensure order
    filled_data = all_slots.join(slot_end_data, on='slot', how='left').sort('slot')

    # Fill missing values for token0 and token1
    filled_data = filled_data.with_columns([
        pl.col('Token0').fill_null(strategy='forward'),
        pl.col('Token1').fill_null(strategy='forward'),
        pl.col('cumulative_inflow').fill_null(strategy='forward'),
        pl.col('cumulative_outflow').fill_null(strategy='forward'),
        pl.col("datetime").fill_null(strategy='backward')
    ])
    return filled_data

def compute_features(data: pl.DataFrame, slot_windows, open_token0, open_token1):
    open_time = data[0, 'datetime']
    open_liquidity = float(open_token0) * open_token1

    features = data.with_columns([
        pl.lit(open_time).alias('open_time'),
        pl.lit(open_token0).alias('open_token0'),
        pl.lit(open_token1).alias('open_token1'),
        pl.lit(open_liquidity).alias('open_liquidity'),
        pl.col('Token0').alias('current_token0'),
        pl.col('Token1').alias('current_token1'),
        ((pl.col('Token0') * pl.col('Token1')) / open_liquidity).alias('current_liquidity_ratio'),
        (pl.col('Token0') / open_token0).alias('current_to_open_token0_ratio'),
        (pl.col('slot') - data[0, 'slot']).alias('slot_elapse'),
        get_time_of_day_feature(pl.col('datetime')).alias('time_of_day')
    ])

    for window in slot_windows:
        token0_shift = pl.col('Token0').shift(window).fill_null(open_token0)  # Shift towards smaller slots
        inflow_shift = pl.col('cumulative_inflow').shift(window).fill_null(0)
        outflow_shift = pl.col('cumulative_outflow').shift(window).fill_null(0)

        features = features.with_columns([
            token0_shift.alias(f'token0_value_{window}slots'),
            (token0_shift / open_token0).alias(f'token0_relative_value_{window}slots'),
            (pl.col('Token0') - token0_shift).alias(f'token0_diff_value_{window}slots'),
            ((pl.col('Token0') - token0_shift) / open_token0).alias(f'token0_relative_diff_value_{window}slots'),
            (pl.col('cumulative_inflow') - inflow_shift).alias(f'inflow_{window}slots'),
            (pl.col('cumulative_outflow') - outflow_shift).alias(f'outflow_{window}slots')
        ])
    
    for i in range(len(slot_windows) - 1):
        features = features.with_columns((2 * pl.col(f"inflow_{slot_windows[i]}slots") - pl.col(f"inflow_{slot_windows[i+1]}slots")).alias(f"inflow_diff_{slot_windows[i]}slots")) 
    features = features.drop("datetime")
    features = features.drop("open_time")
    return features

def update_holding_distribution(holding_distribution, net_delta):
    address = net_delta['From']
    delta = net_delta['cumulative_delta0']
    
    if address in holding_distribution:
        holding_distribution[address] += delta
    else:
        holding_distribution[address] = delta
    
    return holding_distribution

def compute_holding_features(holding_distribution):
    positive_holdings = [v for v in holding_distribution.values() if v > 0]
    negative_holdings = [v for v in holding_distribution.values() if v < 0]
    
    total_token0 = sum(positive_holdings)
    address_proportions = [v / total_token0 for v in positive_holdings]
    
    negative_holdings_sum = sum(negative_holdings)
    num_addresses = len(positive_holdings)
    max_address_holding = max(address_proportions) if address_proportions else 0
    top_5_address_holding = sum(sorted(address_proportions, reverse=True)[:5])
    top_10_address_holding = sum(sorted(address_proportions, reverse=True)[:10])
    holding_entropy_value = entropy(address_proportions) if address_proportions else 0
    
    return {
        'negative_holdings': negative_holdings_sum,
        'num_addresses': num_addresses,
        'max_address_holding': max_address_holding,
        'top_5_address_holding': top_5_address_holding,
        'top_10_address_holding': top_10_address_holding,
        'holding_entropy': holding_entropy_value
    }

def compute_targets(data, slot_windows, open_token0):
    targets = data.with_columns([
        pl.col('Token0').alias('current_token0')
    ])

    for window in slot_windows:
        token0_shift = pl.col('Token0').shift(-window).fill_null(strategy="forward").fill_null(open_token0)
        targets = targets.with_columns([
            token0_shift.alias(f'token0_value_{window}slots')
        ])
        drop_thresholds = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        for threshold in drop_thresholds:
            targets = targets.with_columns([
                pl.when(token0_shift < (pl.col('current_token0') * (1 - threshold))).then(1).otherwise(0).alias(f'token0_drop_{int(threshold*100)}%_{window}slots'),
                token0_shift.alias(f"token0_{window}slots"),
                (token0_shift / pl.col('Token0') - 1).alias(f"token0_changeratio_{window}slots")
            ])

    return targets

def read_and_process(file_path, date):
    print(f"start to process {file_path}")
    try:
        df = pl.read_parquet(file_path)
        # Ensure columns have appropriate data types
        df = df.with_columns([
            pl.col('slot').cast(pl.Int64),
            pl.col('Delta0').cast(pl.Float64),
            pl.col('Delta1').cast(pl.Float64),
            pl.col('Token0').cast(pl.Float64),
            pl.col('Token1').cast(pl.Float64)
        ])
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None, None

    if df.is_empty() or 'slot' not in df.columns:
        print(f"File {file_path} is empty or does not contain 'slot' field.")
        return None, None

    file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
    open_token0 = df[0, 'Token0']
    open_token1 = df[0, 'Token1']
    original_slots = df['slot']
    processed_data, cumulative_data = process_transaction_data(df, file_creation_time)
    all_slots = pl.DataFrame({'slot': range(processed_data['slot'].min(), processed_data['slot'].max() + 1)})
    filled_data = fill_missing_slots(processed_data, all_slots)
    
    all_features = compute_features(filled_data, [15, 30, 60, 120, 240, 480, 960, 1920, 3840, 7680], open_token0, open_token1)
    
    holding_distribution = defaultdict(float)
    holding_features_list = []

    prev_slot = 0
    for row in cumulative_data.iter_rows(named=True):
        current_slot = row['slot']
        if prev_slot != current_slot and prev_slot != 0:
            # Compute holding features
            holding_features = compute_holding_features(holding_distribution)
            holding_features['slot'] = prev_slot  # Add slot information
            holding_features_list.append(holding_features)
        prev_slot = current_slot
        current_net_delta = row['cumulative_delta0']
        
        # Update holding distribution
        holding_distribution = update_holding_distribution(holding_distribution, {'From': row['From'], 'cumulative_delta0': current_net_delta})

    # Compute holding features
    holding_features = compute_holding_features(holding_distribution)
    holding_features['slot'] = current_slot  # Add slot information
    holding_features_list.append(holding_features)

    # Combine holding features into a DataFrame
    holding_features_df = pl.DataFrame(holding_features_list)
    # Merge holding features with other features
    all_features = all_features.join(holding_features_df, on='slot', how='outer')

    all_features = all_features.fill_null(strategy="forward") # for holding feature

    # Compute differences for holding features
    for window in [15, 30, 60, 120, 240, 480, 960, 1920, 3840, 7680]:
        for col in ['top_5_address_holding', 'top_10_address_holding', 'max_address_holding']:
            shift_col = pl.col(col).shift(window).fill_null(1) # avoid big difference
            all_features = all_features.with_columns([
                (pl.col(col) - shift_col).alias(f'{col}_diff_{window}slots')
            ])
        col = 'holding_entropy'
        shift_col = pl.col(col).shift(window).fill_null(0) # avoid big difference
        all_features = all_features.with_columns([
                (pl.col(col) - shift_col).alias(f'{col}_diff_{window}slots')
            ])
    all_targets = compute_targets(filled_data, [15, 30, 60, 120, 240, 480, 960, 1920, 3840, 7680], open_token0)

    # Separate time_of_day column from the rest
    time_of_day_col = all_features.select('time_of_day')
    features_df = all_features.drop('time_of_day')

    # Ensure that all features are stored as float64 to avoid overflow issues
    features_df = features_df.with_columns([
        pl.col(c).cast(pl.Float64)
        for c in features_df.columns
    ])

    # Add time_of_day column back
    features_df = features_df.with_columns(time_of_day_col)
    features_df = features_df.drop('Token0')
    features_df = features_df.drop('Token1')
    # Filter targets_df to keep only original slots
    targets_df = all_targets
    targets_df = targets_df.drop('cumulative_inflow')
    targets_df = targets_df.drop('cumulative_outflow')
    targets_df = targets_df.drop('Token0')
    targets_df = targets_df.drop('Token1')
    targets_df = targets_df.drop('datetime')
    targets_df = targets_df.drop('current_token0')
    return features_df.filter(pl.col('slot').is_in(original_slots)), targets_df.filter(pl.col('slot').is_in(original_slots)), features_df, targets_df, 

def process_file(file_path):
    print(f"Processing file: {file_path}")
    date = os.path.basename(os.path.dirname(file_path))
    try:
        features_df, targets_df, features_allslots, targets_allslots = read_and_process(file_path, date)
    except:
        return
    if features_df is not None and targets_df is not None:
        output_dir = os.path.join("processed_data", date)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        features_df.write_parquet(os.path.join(output_dir, f"{os.path.basename(file_path)}_features.parquet"))
        targets_df.write_parquet(os.path.join(output_dir, f"{os.path.basename(file_path)}_targets.parquet"))
        features_allslots.write_parquet(os.path.join(output_dir, f"{os.path.basename(file_path)}_featuresallslots.parquet"))
        targets_allslots.write_parquet(os.path.join(output_dir, f"{os.path.basename(file_path)}_targetsallslots.parquet"))
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
    root_dir = r"..\coin_data"  # Windows路径
    file_paths = get_all_parquet_files(root_dir)

    # 打印获取到的文件路径
    print(f"Found {len(file_paths)} files to process.")
    for file in file_paths:
        process_file(file)
