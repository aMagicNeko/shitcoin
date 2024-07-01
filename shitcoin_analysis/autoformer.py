from transformers import AutoformerConfig, AutoformerForPrediction, AutoformerModel
import torch
import os
import polars as pl
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset, Dataset
from util import swap_token_amount_base_in
def get_all_parquet_files(root_dir):
    features_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            subdir = os.path.join(dirpath, dirname)
            for subdirpath, subdirnames, subfilenames in os.walk(subdir):
                for file in subfilenames:
                    if file.endswith('featuresallslots.parquet'):
                        features_files.append(os.path.join(subdirpath, file))
    return features_files

def get_sorted_file_list_by_length(features_files):
    file_lengths = []
    for f_file in features_files:
        features_df = pl.read_parquet(f_file)
        file_lengths.append((f_file, len(features_df)))
    
    file_lengths.sort(key=lambda x: x[1])
    sorted_files = [f[0] for f in file_lengths]
    return sorted_files

def compute_target(features_df):
    initial_token0 = 24000000
    initial_token1 = swap_token_amount_base_in(initial_token0, features_df[0, 'current_token0'], features_df[0, 'current_token1'], True)
    target = swap_token_amount_base_in(initial_token1, features_df[:, 'current_token0'], features_df[:, 'current_token1'], False)
    return target.to_numpy()

def compute_scaler_params(features_files, dynamic_feature_columns, static_feature_columns):
    dynamic_mean = None
    dynamic_M2 = None
    static_mean = None
    static_M2 = None
    target_mean = None
    target_M2 = None
    n = 0
    
    for f_file in features_files:
        features_df = pl.read_parquet(f_file)

        # 动态特征
        dynamic_features_df = features_df.select(dynamic_feature_columns)
        new_dynamic_M2 = np.zeros((dynamic_features_df.shape[1]))

        # 静态特征
        static_features_df = features_df.select(static_feature_columns).head(1)  # 假设静态特征在整个文件中相同
        new_static_M2 = np.zeros((static_features_df.shape[1]))

        # Target计算
        target_values = compute_target(features_df)
        new_target_M2 = 0

        # 动态特征均值和方差计算
        if dynamic_mean is None:
            dynamic_mean = dynamic_features_df.mean(axis=0).to_numpy().reshape((-1,))
            for i, col in enumerate(dynamic_features_df.columns):
                new_dynamic_M2[i] = ((dynamic_features_df[col] - dynamic_mean[i]) ** 2).sum()
            n = len(dynamic_features_df)
            dynamic_M2 = new_dynamic_M2
        else:
            new_n = len(dynamic_features_df)
            new_dynamic_mean = dynamic_features_df.mean(axis=0).to_numpy().reshape((-1,))
            for i, col in enumerate(dynamic_features_df.columns):
                new_dynamic_M2[i] = ((dynamic_features_df[col] - dynamic_mean[i]) ** 2).sum()
            
            total_n = n + new_n
            delta_dynamic = new_dynamic_mean - dynamic_mean
            dynamic_mean = (dynamic_mean * n + new_dynamic_mean * new_n) / total_n
            dynamic_M2 += new_dynamic_M2 + delta_dynamic ** 2 * n * new_n / total_n

            n = total_n

        # 静态特征均值和方差计算
        if static_mean is None:
            static_mean = static_features_df.mean(axis=0).to_numpy().reshape((-1,))
            for i, col in enumerate(static_features_df.columns):
                new_static_M2[i] = ((static_features_df[col] - static_mean[i]) ** 2).sum()
            static_M2 = new_static_M2
        else:
            new_static_mean = static_features_df.mean(axis=0).to_numpy().reshape((-1,))
            for i, col in enumerate(static_features_df.columns):
                new_static_M2[i] = ((static_features_df[col] - static_mean[i]) ** 2).sum()
            
            delta_static = new_static_mean - static_mean
            static_mean = (static_mean + new_static_mean) / 2  # 由于静态特征在整个文件中相同，这里直接取平均
            static_M2 += new_static_M2 + delta_static ** 2 / 2  # 由于静态特征在整个文件中相同，这里直接取平均

        # 目标值均值和方差计算
        if target_mean is None:
            target_mean = target_values.mean()
            new_target_M2 = ((target_values - target_mean) ** 2).sum()
            target_M2 = new_target_M2
        else:
            new_n = len(target_values)
            new_target_mean = target_values.mean()
            new_target_M2 = ((target_values - target_mean) ** 2).sum()

            total_n = n + new_n
            delta_target = new_target_mean - target_mean
            target_mean = (target_mean * n + new_target_mean * new_n) / total_n
            target_M2 += new_target_M2 + delta_target ** 2 * n * new_n / total_n

    dynamic_variance = dynamic_M2 / (n - 1)
    dynamic_std = dynamic_variance ** 0.5

    static_variance = static_M2 / (n - 1)
    static_std = static_variance ** 0.5

    target_variance = target_M2 / (n - 1)
    target_std = target_variance ** 0.5

    # 保存标准化参数到 JSON 文件
    scaler_params = {
        'dynamic_mean': dynamic_mean.tolist(),
        'dynamic_std': dynamic_std.tolist(),
        'static_mean': static_mean.tolist(),
        'static_std': static_std.tolist(),
        'target_mean': target_mean,
        'target_std': target_std
    }
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)

    return dynamic_mean, dynamic_std, static_mean, static_std

def load_scaler_params(json_file_path):
    with open(json_file_path, 'r') as f:
        scaler_params = json.load(f)
    
    dynamic_mean = np.array(scaler_params['dynamic_mean'])
    dynamic_std = np.array(scaler_params['dynamic_std'])
    static_mean = np.array(scaler_params['static_mean'])
    static_std = np.array(scaler_params['static_std'])
    target_mean = scaler_params['target_mean']
    target_std = scaler_params['target_std']
    return dynamic_mean, dynamic_std, static_mean, static_std, target_mean, target_std

def standardize_data(features_df, dynamic_feature_columns, static_feature_columns, dynamic_mean, dynamic_std, static_mean, static_std, target_mean, target_std):
    # Standardize dynamic features
    dynamic_features_df = features_df.select(dynamic_feature_columns)
    for i, col in enumerate(dynamic_feature_columns):
        dynamic_features_df = dynamic_features_df.with_columns((pl.col(col) - dynamic_mean[i]) / dynamic_std[i])

    # Standardize static features
    static_features_df = features_df.select(static_feature_columns)
    for i, col in enumerate(static_feature_columns):
        static_features_df = static_features_df.with_columns((pl.col(col) - static_mean[i]) / static_std[i])

    # Combine original features and differenced terms
    combined_features_df = pl.concat([dynamic_features_df, features_df.select(['slot_elapse']), static_features_df], how='horizontal')
    
    target_df = compute_target(features_df)
    target_df = (target_df - target_mean) / target_std
    return combined_features_df, target_df

from torch.nn.utils.rnn import pad_sequence

def pad_and_save_batches(sorted_files, dynamic_feature_columns, static_feature_columns, dynamic_mean, dynamic_std, static_mean, static_std, target_mean, target_std, batch_size=256):
    batch_dir = 'padded_batch_data'
    os.makedirs(batch_dir, exist_ok=True)
    batch_index = 0

    for i in range(0, len(sorted_files), batch_size):
        batch_files = sorted_files[i:i+batch_size]
        batch_features = []
        batch_targets = []
        
        for f_file in batch_files:
            features_df = pl.read_parquet(f_file)
            standardized_features_df, target_df = standardize_data(features_df, dynamic_feature_columns, static_feature_columns, dynamic_mean, dynamic_std, static_mean, static_std, target_mean, target_std)
            batch_features.append(torch.tensor(standardized_features_df.to_numpy(), dtype=torch.float32))
            batch_targets.append(torch.tensor(target_df, dtype=torch.float32))

        # Pad sequences to the length of the longest sequence in the batch
        padded_features = pad_sequence(batch_features, batch_first=True, padding_value=0)
        padded_targets = pad_sequence(batch_targets, batch_first=True, padding_value=0)

        batch_file = os.path.join(batch_dir, f'batch_{batch_index}.pt')
        torch.save((padded_features, padded_targets), batch_file)
        batch_index += 1

    return [os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.endswith('.pt')]

from torch.utils.data import Dataset, DataLoader

class BatchDataset(Dataset):
    def __init__(self, batch_files):
        self.batch_files = batch_files

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        batch_file = self.batch_files[idx]
        features, target = torch.load(batch_file)
        return features, target

def train_model(model, batch_files, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        j = 0
        for file in batch_files:
            features, targets = torch.load(file)
            features = features.to(device)
            targets = targets.squeeze(2).to(device)
            optimizer.zero_grad()
            batch_size = features.size(0)
            past_values = torch.zeros((batch_size, context_length + lags_length), device=device)
            past_time_features = torch.zeros((batch_size, context_length + lags_length, len(dynamic_feature_col)+len(time_col)), device=device)
            for i in range(features.size(1) - prediction_length):  # Iterate over time steps
                available_data_length = min(i+1, context_length)
                if available_data_length > 0:
                    past_values[:, :available_data_length] = targets[:batch_size, i-available_data_length+1:i+1]
                    past_time_features[:, :available_data_length] = features[:batch_size, i-available_data_length+1:i+1, :len(dynamic_feature_col)+len(time_col)]
                future_values = targets[:batch_size, i+1:i+prediction_length+1]
                future_time_features = features[:batch_size, i+1:i+prediction_length+1, :len(dynamic_feature_col)+len(time_col)]
                static_real_features = features[:batch_size, 0, len(dynamic_feature_col)+len(time_col):]
                output = model(
                    past_values=past_values,
                    past_time_features=past_time_features,
                    static_real_features=static_real_features,
                    past_observed_mask=torch.ones(past_values.shape, device=device),
                    future_values=future_values,
                    future_time_features=future_time_features,
                    use_cache=True
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                output.loss.backward()
                optimizer.step()
                running_loss += output.loss.item()
                print(i)
            save_model(model, optimizer, epoch, running_loss, path=f'model_epoch_{epoch}_{j}.pth')
            j+=1
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(batch_files):.4f}")
        save_model(model, optimizer, epoch, running_loss, path=f'model_epoch_{epoch}.pth')

def save_model(model, optimizer, epoch, loss, path='model.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model(model, optimizer, path='model.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def new_loss_function(predictions, initial_token0, has_sold, final_token0):
    if has_sold:
        profit = final_token0 - initial_token0
    else:
        profit = -initial_token0
    # Loss should be negative profit to maximize profit during optimization
    loss = -profit
    return loss

dynamic_feature_col = ['current_token0', 'current_token1', 'cumulative_inflow', 'cumulative_outflow', 'negative_holdings', 'num_addresses', 'max_address_holding', 'top_5_address_holding', 'top_10_address_holding', 'holding_entropy']
# diff it
time_col = ['slot_elapse']
static_feature_col = ['open_token0', 'open_token1']
context_length = 300
lags_length = 7
input_size = 1
prediction_length=5
batch_dir = 'padded_batch_data'
if __name__ == "__main__":
    root_dir = 'processed_data'
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_files = get_all_parquet_files(root_dir)
    compute_scaler_params(features_files, dynamic_feature_col, static_feature_col)
    dynamic_mean, dynamic_std, static_mean, static_std, target_mean, target_std = load_scaler_params('scaler_params.json')
    sorted_files = get_sorted_file_list_by_length(features_files)
    batch_files = pad_and_save_batches(sorted_files, dynamic_feature_col, static_feature_col, dynamic_mean, dynamic_std, static_mean, static_std, target_mean, target_std, batch_size=200)
    batch_files = [os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.endswith('.pt')]
    # Define the model configuration and instantiate the model
    config = AutoformerConfig(
        prediction_length=5, # 2s prediction
        context_length=context_length,  # Use the past 3000 time steps as context
        input_size=1,  # Adjust input size, target
        lags_sequence=[1, 2, 3, 4, 5, 6, 7],  # Lags sequence
        num_time_features=len(time_col),  # Number of time features
        num_dynamic_real_features=len(dynamic_feature_col),  # Number of dynamic real valued features
        num_static_categorical_features=0,  # Number of static categorical features
        num_static_real_features=len(static_feature_col),  # Number of static real features
        d_model=64,
        encoder_layers=4,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        dropout=0.1,
        moving_average=25,
        autocorrelation_factor=3,
        use_cache=True
    )

    model = AutoformerForPrediction(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Compute scaler parameters
    #compute_scaler_params(features_files, dynamic_feature_col, static_feature_col)
    
    # Train the model
    train_model(model, batch_files, optimizer, device, num_epochs=10)

    # Load the trained model
    #model, optimizer, epoch, loss = load_model(model, optimizer, path='final_model.pth')
