import os
import random
import polars as pl
from polars import col
import json
import numpy as np

def read_and_process_polars(file_path):
    try:
        # Read the Parquet file using Polars
        df = pl.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

def preprocess_data(features_df, targets_df, feature_columns, target_columns, save_params=False, params_file_path=None):
    # Handle missing values
    features_df = features_df.fill_null(strategy="forward").fill_null(strategy="backward")
    targets_df = targets_df.fill_null(strategy="forward").fill_null(strategy="backward")
    # Manually one-hot encode the 'time_of_day' feature
    #time_of_day_categories = ['morning', 'afternoon', 'evening', 'night']
    #for category in time_of_day_categories:
    #   features_df = features_df.with_columns((col("time_of_day") == category).cast(pl.Int8).alias(f"time_of_day_{category}"))
    # Remove non-numeric columns from normalization
    non_numeric_columns = ['time_of_day']
    numeric_feature_columns = [col for col in feature_columns if col not in non_numeric_columns]
    
    params = {}
    
    # Normalize the data
    for column in numeric_feature_columns:
        mean = features_df[column].mean()
        std = features_df[column].std()
        features_df = features_df.with_columns(((col(column) - mean) / std).alias(column))
        params[column] = {'mean': mean, 'std': std}
    for column in target_columns:
        if "value" in column:
            mean = targets_df[column].mean()
            std = targets_df[column].std()
            targets_df = targets_df.with_columns(((col(column) - mean) / std).alias(column))
            params[column] = {'mean': mean, 'std': std}
    
    # Remove unnecessary columns at the end
    features_df = features_df.drop(non_numeric_columns)
    targets_df = targets_df.select(target_columns)
    targets_df = targets_df.drop('slot')
    
    if save_params and params_file_path:
        save_standardization_params(params, params_file_path)
    
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

def save_standardization_params(params, file_path):
    with open(file_path, 'w') as f:
        json.dump(params, f)

if __name__ == "__main__":
    feature, target = process_files_randomly('processed_data', 4000)
    feature, target = preprocess_data(feature, target, feature.columns, target.columns, "standardization_params.json")