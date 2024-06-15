import os
import random
import polars as pl
from polars import col
from multiprocessing import Pool

def read_and_process_polars(file_path):
    try:
        # Read the Parquet file using Polars
        df = pl.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

def preprocess_data(features_df, targets_df, feature_columns, target_columns):
    # Handle missing values
    features_df = features_df.fill_null(strategy="forward").fill_null(strategy="backward")
    targets_df = targets_df.fill_null(strategy="forward").fill_null(strategy="backward")

    # Manually one-hot encode the 'time_of_day' feature
    time_of_day_categories = ['morning', 'afternoon', 'evening', 'night']
    for category in time_of_day_categories:
        features_df = features_df.with_columns((col("time_of_day") == category).cast(pl.Int8).alias(f"time_of_day_{category}"))

    # Remove non-numeric columns from normalization
    non_numeric_columns = ["open_time", 'time_of_day']
    numeric_feature_columns = [col for col in feature_columns if col not in non_numeric_columns]
    
    # Normalize the data
    for column in numeric_feature_columns:
        mean = features_df[column].mean()
        std = features_df[column].std()
        features_df = features_df.with_columns(((col(column) - mean) / std).alias(column))
    
    # Normalize elapsed_time_since_open
    #min_time = features_df["slot_elapse"].min()
    #max_time = features_df["slot_elapse"].max()
    #features_df = features_df.with_columns(((col("slot_elapse") - min_time) / (max_time - min_time)).alias("slot_elapse"))

    for column in target_columns:
        if "value" in column:
            mean = targets_df[column].mean()
            std = targets_df[column].std()
            targets_df = targets_df.with_columns(((col(column) - mean) / std).alias(column))
    # Remove unnecessary columns at the end
    features_df = features_df.drop(non_numeric_columns)
    targets_df = targets_df.select(target_columns)

    return features_df, targets_df

def process_files_randomly(directory, num_files):
    # Get a list of all feature and target files in the directory
    feature_files = [os.path.join(root, file)
                     for root, dirs, files in os.walk(directory)
                     for file in files if file.endswith("_features.parquet")]
    target_files = [os.path.join(root, file)
                    for root, dirs, files in os.walk(directory)
                    for file in files if file.endswith("_targets.parquet")]

    # Ensure we have pairs of feature and target files
    file_pairs = [(f, f.replace("_features.parquet", "_targets.parquet")) for f in feature_files if f.replace("_features.parquet", "_targets.parquet") in target_files]
    
    # Select a random subset of file pairs
    selected_pairs = random.sample(file_pairs, min(num_files, len(file_pairs)))
    
    # Read and process the selected pairs of files
    feature_dataframes = []
    target_dataframes = []
    for feature_file, target_file in selected_pairs:
        feature_df = read_and_process_polars(feature_file)
        target_df = read_and_process_polars(target_file)
        if feature_df is not None and target_df is not None:
            feature_dataframes.append(feature_df)
            target_dataframes.append(target_df)
    
    # Concatenate all dataframes
    if feature_dataframes and target_dataframes:
        combined_features_df = pl.concat(feature_dataframes)
        combined_targets_df = pl.concat(target_dataframes)
        return combined_features_df, combined_targets_df
    else:
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Read and process random subset of feature and target files using Polars.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing the feature and target files.")
    parser.add_argument("--num_files", type=int, default=100, help="Number of random file pairs to read and process.")
    
    args = parser.parse_args()
    
    feature_columns = [
        'open_token0', 'current_token0', 'token0_value_15slots', 'token0_value_30slots', 
        'token0_value_90slots', 'token0_value_180slots', 'token0_value_360slots', 'token0_value_900slots', 
        'token0_value_1800slots', 'token0_value_3600slots', 'token0_value_5400slots', 'token0_value_10800slots', 
        'inflow_15slots', 'outflow_15slots', 'inflow_30slots', 'outflow_30slots', 'inflow_90slots', 'outflow_90slots', 
        'inflow_180slots', 'outflow_180slots', 'inflow_360slots', 'outflow_360slots', 'inflow_900slots', 'outflow_900slots', 
        'inflow_1800slots', 'outflow_1800slots', 'inflow_3600slots', 'outflow_3600slots', 'inflow_5400slots', 'outflow_5400slots', 
        'inflow_10800slots', 'outflow_10800slots', 'negative_holdings', 'num_addresses', 'max_address_holding', 
        'top_5_address_holding', 'top_10_address_holding', 'holding_entropy', 'elapsed_time_since_open', 
        'time_of_day_morning', 'time_of_day_afternoon', 'time_of_day_evening', 'time_of_day_night'
    ]
    
    target_columns = [
        'token0_value_15slots', 'token0_value_30slots', 'token0_value_90slots', 'token0_value_180slots',
        'token0_value_900slots', 'token0_value_1800slots', 'token0_value_5400slots', 'token0_value_10800slots'
    ]

    combined_features_df, combined_targets_df = process_files_randomly(args.directory, args.num_files)
    
    if combined_features_df is not None and combined_targets_df is not None:
        features_normalized_df, targets_normalized_df = preprocess_data(combined_features_df, combined_targets_df, feature_columns, target_columns)
        print(features_normalized_df)
        print(targets_normalized_df)
    else:
        print("No files were processed.")
