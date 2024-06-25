from transformers import AutoformerConfig, AutoformerForPrediction, AutoformerModel
import torch
import os
import polars as pl
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
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

def compute_scaler_params(features_files, dynamic_feature_columns, static_feature_columns):
    dynamic_mean = None
    dynamic_M2 = None
    static_mean = None
    static_M2 = None
    n = 0
    
    for f_file in features_files:
        features_df = pl.read_parquet(f_file)

        # 动态特征
        dynamic_features_df = features_df.select(dynamic_feature_columns)
        new_dynamic_M2 = np.zeros((dynamic_features_df.shape[1]))

        # 静态特征
        static_features_df = features_df.select(static_feature_columns).head(1)  # 假设静态特征在整个文件中相同
        new_static_M2 = np.zeros((static_features_df.shape[1]))

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

    dynamic_variance = dynamic_M2 / (n - 1)
    dynamic_std = dynamic_variance ** 0.5

    static_variance = static_M2 / (n - 1)
    static_std = static_variance ** 0.5

    # 保存标准化参数到 JSON 文件
    scaler_params = {
        'dynamic_mean': dynamic_mean.tolist(),
        'dynamic_std': dynamic_std.tolist(),
        'static_mean': static_mean.tolist(),
        'static_std': static_std.tolist()
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
    
    return dynamic_mean, dynamic_std, static_mean, static_std

def standardize_data(features_df, dynamic_feature_columns, static_feature_columns, dynamic_mean, dynamic_std, static_mean, static_std):
    # Standardize dynamic features
    dynamic_features_df = features_df.select(dynamic_feature_columns)
    standardized_dynamic_features_df = (dynamic_features_df - dynamic_mean) / dynamic_std

    # Standardize static features
    static_features_df = features_df.select(static_feature_columns)
    standardized_static_features_df = (static_features_df - static_mean) / static_std

    # Combine original features and differenced terms
    combined_features_df = pl.concat([standardized_dynamic_features_df, features_df.select(['slot_elapse']), standardized_static_features_df], axis=1)
    
    return combined_features_df

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

class SequenceDataset(Dataset):
    def __init__(self, features_files, dynamic_feature_columns, static_feature_columns, dynamic_mean, dynamic_std, dynamic_diff_mean, dynamic_diff_std, static_mean, static_std):
        self.features_files = features_files
        self.dynamic_feature_columns = dynamic_feature_columns
        self.static_feature_columns = static_feature_columns
        self.dynamic_mean = dynamic_mean
        self.dynamic_std = dynamic_std
        self.dynamic_diff_mean = dynamic_diff_mean
        self.dynamic_diff_std = dynamic_diff_std
        self.static_mean = static_mean
        self.static_std = static_std

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        f_file = self.features_files[idx]
        features_df = pl.read_parquet(f_file)
        
        # Standardize features
        standardized_features_df = standardize_data(features_df, self.dynamic_feature_columns, self.static_feature_columns, self.dynamic_mean, self.dynamic_std, self.dynamic_diff_mean, self.dynamic_diff_std, self.static_mean, self.static_std)
        standardized_features = torch.tensor(standardized_features_df.to_numpy(), dtype=torch.float32)
        return standardized_features

def train_model(model, features_files, optimizer, device, num_epochs=10):
    dynamic_feature_col = ['current_token0', 'current_token1', 'cumulative_inflow', 'cumulative_outflow', 'negative_holdings', 'num_addresses', 'max_address_holding', 'top_5_address_holding', 'top_10_address_holding', 'holding_entropy']
    static_feature_col = ['open_token0', 'open_token1']
    time_col = ['slot_elapse']
    
    dynamic_mean, dynamic_std, static_mean, static_std = load_scaler_params('scaler_params.json')
    
    dataset = SequenceDataset(features_files, dynamic_feature_col, static_feature_col, dynamic_mean, dynamic_std, static_mean, static_std)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    criterion = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for features in loader:
            features = features.squeeze(0).to(device)  # Remove the batch dimension
            initial_token0 = 24000000
            initial_token1 = swap_token_amount_base_in(initial_token0, features[0, 0], features[0, 1], True)
            target = swap_token_amount_base_in(initial_token1, features[:, 0], features[:, 1], False)
            optimizer.zero_grad()
            mems = None
            
            for i in range(features.size(0) - 7):  # Iterate over time steps
                past_values = torch.zeros((context_length + lags_length,))
                past_time_features = torch.zeros(context_length + lags_length, len(dynamic_feature_col)+len(time_col))
                available_data_length = min(i, context_length)
                if available_data_length > 0:
                    past_values[:available_data_length] = target[i-available_data_length:i]
                    past_time_features[:available_data_length] = features[i-available_data_length:i, :len(dynamic_feature_col)+len(time_col)]
                static_real_features = features[len(dynamic_feature_col)+len(time_col):]
                output = model.forward(
                    past_values=past_values.unsqueeze(0),
                    past_time_features=past_time_features.unsqueeze(0),
                    static_real_features=static_real_features.unsqueeze(0)
                )
                print(output)
                loss = criterion(output[:, :, 5], target.unsqueeze(0))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(loader):.4f}")
        save_model(model, optimizer, epoch, running_loss, path=f'model_epoch_{epoch}.pth')

dynamic_feature_col = ['current_token0', 'current_token1', 'cumulative_inflow', 'cumulative_outflow', 'negative_holdings', 'num_addresses', 'max_address_holding', 'top_5_address_holding', 'top_10_address_holding', 'holding_entropy']
# diff it
time_col = ['slot_elapse']
static_feature_col = ['open_token0', 'open_token1']
context_length = 3000
lags_length = 7
input_size = 1
if __name__ == "__main__":
    root_dir = 'processed'
    features_files = get_all_parquet_files(root_dir)

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
        encoder_layers=3,
        decoder_layers=3,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        dropout=0.1,
        moving_average=25,
        autocorrelation_factor=3,
        d_model=32  # Dimension of the model
    )

    model = AutoformerModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Compute scaler parameters
    compute_scaler_params(features_files, dynamic_feature_col, static_feature_col)
    
    # Train the model
    train_model(model, features_files, optimizer, device, num_epochs=10)

    # Load the trained model
    model, optimizer, epoch, loss = load_model(model, optimizer, path='final_model.pth')
