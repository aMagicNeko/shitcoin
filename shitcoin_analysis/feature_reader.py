import os
import glob
import random
import polars as pl
from polars import col
from multiprocessing import Pool

def read_and_process_polars(file_path):
    try:
        # Read the CSV file using Polars
        df = pl.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

def preprocess_data(features_df, targets_df, feature_columns, target_columns):
    # Handle missing values
    features_df = features_df.fill_null(strategy="forward").fill_null(strategy="backward")
    targets_df = targets_df.fill_null(strategy="forward").fill_null(strategy="backward")

    # Add time-related features
    def time_of_day(hour):
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"

    # Check if 'current_time' and 'open_time' columns are present
    if "current_time" not in features_df.columns or "open_time" not in features_df.columns:
        raise ValueError("Required columns 'current_time' or 'open_time' are missing.")

    features_df = features_df.with_columns([
        col("current_time").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S").dt.hour().apply(time_of_day).alias("time_of_day"),
        (col("current_time").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S") - col("open_time").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S")).dt.seconds().alias("elapsed_time_since_open")
    ])

    # Manually one-hot encode the 'time_of_day' feature
    time_of_day_categories = ['morning', 'afternoon', 'evening', 'night']
    for category in time_of_day_categories:
        features_df = features_df.with_columns((col("time_of_day") == category).cast(pl.Int8).alias(f"time_of_day_{category}"))

    # Remove the original 'time_of_day' column
    features_df = features_df.drop("time_of_day")

    # Remove non-numeric columns from normalization
    non_numeric_columns = ["open_time", "current_time"]
    numeric_feature_columns = [col for col in feature_columns if col not in non_numeric_columns]
    
    # Normalize the data
    for column in numeric_feature_columns:
        mean = features_df[column].mean()
        std = features_df[column].std()
        features_df = features_df.with_columns(((col(column) - mean) / std).alias(column))
    
    # Normalize elapsed_time_since_open
    min_time = features_df["elapsed_time_since_open"].min()
    max_time = features_df["elapsed_time_since_open"].max()
    features_df = features_df.with_columns(((col("elapsed_time_since_open") - min_time) / (max_time - min_time)).alias("elapsed_time_since_open"))

    for column in target_columns:
        mean = targets_df[column].mean()
        std = targets_df[column].std()
        targets_df = targets_df.with_columns(((col(column) - mean) / std).alias(column))

    # Remove unnecessary columns at the end
    features_df = features_df.drop(non_numeric_columns)
    #features_df = features_df.select(numeric_feature_columns)
    targets_df = targets_df.select(target_columns)

    return features_df, targets_df

def process_files_randomly(directory, num_files):
    # Get a list of all feature and target files in the directory
    feature_files = glob.glob(os.path.join(directory, "*_features.csv"))
    target_files = glob.glob(os.path.join(directory, "*_targets.csv"))

    # Ensure we have pairs of feature and target files
    file_pairs = [(f, f.replace("_features.csv", "_targets.csv")) for f in feature_files if f.replace("_features.csv", "_targets.csv") in target_files]
    
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
        'open_token0', 'current_token0', 'token0_value_5s', 'token0_value_10s', 
        'token0_value_30s', 'token0_value_60s', 'token0_value_120s', 'token0_value_300s', 'token0_value_600s', 
        'token0_value_1200s', 'token0_value_1800s', 'token0_value_3600s', 'inflow_5s', 'outflow_5s', 
        'inflow_10s', 'outflow_10s', 'inflow_30s', 'outflow_30s', 'inflow_60s', 'outflow_60s', 'inflow_120s', 
        'outflow_120s', 'inflow_300s', 'outflow_300s', 'inflow_600s', 'outflow_600s', 'inflow_1200s', 
        'outflow_1200s', 'inflow_1800s', 'outflow_1800s', 'inflow_3600s', 'outflow_3600s', 'negative_holdings', 
        'num_addresses', 'max_address_holding', 'top_5_address_holding', 'top_10_address_holding', 
        'holding_entropy', 'elapsed_time_since_open', 'time_of_day_morning', 'time_of_day_afternoon', 'time_of_day_evening', 'time_of_day_night'
    ]
    
    target_columns = [
        'token0_value_5s', 'token0_value_10s', 'token0_value_30s', 'token0_value_60s',
        'token0_value_300s', 'token0_value_600s', 'token0_value_1800s', 'token0_value_3600s'
    ]

    combined_features_df, combined_targets_df = process_files_randomly(args.directory, args.num_files)
    
    if combined_features_df is not None and combined_targets_df is not None:
        features_normalized_df, targets_normalized_df = preprocess_data(combined_features_df, combined_targets_df, feature_columns, target_columns)
        print(features_normalized_df)
        print(targets_normalized_df)
    else:
        print("No files were processed.")
