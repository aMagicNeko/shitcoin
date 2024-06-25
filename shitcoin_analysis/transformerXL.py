import polars as pl
import numpy as np
import json
import torch.nn as nn
import math
from transformers import TransfoXLConfig, TransfoXLModel
import torch
from torch.utils.data import Dataset, DataLoader
import os
from util import swap_token_amount_base_in
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

class ChunkedTransactionDataset(Dataset):
    def __init__(self, features_files, targets_files, dynamic_mean, dynamic_std, static_mean, static_std, dynamic_feature_columns, static_feature_columns):
        self.features_files = features_files
        self.targets_files = targets_files
        self.dynamic_mean = dynamic_mean
        self.dynamic_std = dynamic_std
        self.static_mean = static_mean
        self.static_std = static_std
        self.dynamic_feature_columns = dynamic_feature_columns
        self.static_feature_columns = static_feature_columns

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        f_file = self.features_files[idx]
        t_file = self.targets_files[idx]

        features_df = pl.read_parquet(f_file)
        targets_df = pl.read_parquet(t_file)

        # 跳过行数为0的文件
        if features_df.shape[0] == 0 or targets_df.shape[0] == 0:
            return self.__getitem__((idx + 1) % len(self))  # 递归调用获取下一个文件

        # 标准化动态特征
        dynamic_features_df = features_df.select(self.dynamic_feature_columns)
        for i, col in enumerate(self.dynamic_feature_columns):
            dynamic_features_df = dynamic_features_df.with_columns((pl.col(col) - self.dynamic_mean[i]) / self.dynamic_std[i])

        # 标准化静态特征
        static_features_df = features_df.select(self.static_feature_columns).head(1)  # 假设静态特征在整个文件中相同
        for j, col in enumerate(self.static_feature_columns):
            static_features_df = static_features_df.with_columns((pl.col(col) - self.static_mean[j]) / self.static_std[j])

        dynamic_features_tensor = torch.tensor(dynamic_features_df.to_numpy(), dtype=torch.float32)
        static_features_tensor = torch.tensor(static_features_df.to_numpy(), dtype=torch.float32).repeat(dynamic_features_tensor.size(0), 1)

        # 结合动态和静态特征
        features_tensor = torch.cat((dynamic_features_tensor, static_features_tensor), dim=1)
        #targets_tensor = torch.tensor(targets_df[self.target_column].to_numpy(), dtype=torch.float32).reshape((-1, 1))

        return features_tensor

class TransformerXLModel(nn.Module):
    def __init__(self, input_dim, static_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, max_len=100):
        super(TransformerXLModel, self).__init__()
        self.embedding = nn.Linear(input_dim - static_dim, d_model)  # 动态特征嵌入
        self.config = TransfoXLConfig(d_model=d_model, n_head=nhead, d_inner=dim_feedforward, n_layer=num_layers, dropout=dropout)
        self.transformer_xl = TransfoXLModel(self.config)
        self.fc_static = nn.Linear(static_dim, d_model)  # 静态特征嵌入
        self.fc_out = nn.Linear(d_model * 2, 1)  # 合并动态和静态特征
        self.d_model = d_model
        self.static_dim = static_dim

    def forward(self, src, mems=None):
        dynamic_features = src[:, :-self.static_dim] # batch, dim
        static_features = src[:, -self.static_dim:]

        dynamic_features = self.embedding(dynamic_features) * math.sqrt(self.d_model)
        combined_features = torch.cat((dynamic_features, static_features), dim=-1)

        output, mems = self.transformer_xl(inputs_embeds=combined_features, mems=mems)
        output = self.fc_out(output)
        return torch.sigmoid(output), mems

def get_all_parquet_files(root_dir):
    features_files = []
    targets_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            subdir = os.path.join(dirpath, dirname)
            for subdirpath, subdirnames, subfilenames in os.walk(subdir):
                for file in subfilenames:
                    if file.endswith('featuresallslots.parquet'):
                        features_files.append(os.path.join(subdirpath, file))
                    elif file.endswith('targetsallslots.parquet'):
                        targets_files.append(os.path.join(subdirpath, file))
    return features_files, targets_files

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
    # 损失应当是收益的负值，这样在优化时会最大化收益
    loss = -profit
    return loss

def train_model(model, train_loader, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features in train_loader:
            features = features.to(device) # batch, slot, dim
            #features = features.squeeze()
            print(features.shape)
            initial_token0 = 24000000
            has_sold = False
            token1 = swap_token_amount_base_in(initial_token0, features[0, 0, 0], features[0, 0, 1], True)
            final_token0 = 0
            optimizer.zero_grad()
            mems = None
            
            for i in range(features.size(1)):
                slot_features = features[:, i, :]
                outputs, mems = model(slot_features, mems=mems)
                signal = outputs.squeeze().item()
                if signal > 0.5:
                    final_token0 = swap_token_amount_base_in(token1, slot_features[0, 0], slot_features[0, 1], False)
                    has_sold = True
                    break
            loss = new_loss_function(outputs, initial_token0, has_sold, final_token0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        save_model(model, optimizer, epoch, running_loss, path=f'model_epoch_{epoch}.pth')

# 示例代码
if __name__ == "__main__":
    root_dir = 'processed_data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_files, targets_files = get_all_parquet_files(root_dir)

    dynamic_feature_columns = ['current_token0', 'current_token1', 'slot_elapse', 'inflow_15slots', 'outflow_15slots', 'inflow_diff_15slots', 'outflow_diff_15slots', 'negative_holdings', 'negative_holdings_diff_15slots', 'num_addresses', 'num_addresses_diff_15slots', 'max_address_holding', 'max_address_holding_diff_15slots', 'top_5_address_holding', 'top_5_address_holding_diff_15slots', 'top_10_address_holding', 'top_10_address_holding_diff_15slots', 'holding_entropy', 'holding_entropy_diff_15slots']
    static_feature_columns = ['open_token0', 'open_token1']

    #dynamic_mean, dynamic_std, static_mean, static_std = compute_scaler_params(features_files, dynamic_feature_columns, static_feature_columns)
    dynamic_mean, dynamic_std, static_mean, static_std = load_scaler_params('scaler_params.json')
    train_dataset = ChunkedTransactionDataset(features_files, targets_files, dynamic_mean, dynamic_std, static_mean, static_std, dynamic_feature_columns, static_feature_columns, target_column)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    input_dim = len(dynamic_feature_columns) + len(static_feature_columns)  # 动态和静态特征的总数
    model = TransformerXLModel(input_dim=input_dim, static_dim=len(static_feature_columns), d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    train_model(model, train_loader, optimizer, device, num_epochs)
