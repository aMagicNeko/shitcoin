import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import os
import polars as pl
import json
import numpy as np
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self.generate_positional_encoding(d_model, max_len)

    def generate_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        if x.size(1) > self.max_len:
            self.pe = self.generate_positional_encoding(self.d_model, x.size(1))
            self.max_len = x.size(1)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.d_model = d_model

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.pos_decoder = PositionalEncoding(d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask, src_mask, tgt_key_padding_mask, src_key_padding_mask)
        output = self.fc_out(output.mean(dim=0))  # 平均池化
        return torch.sigmoid(output)

class TransactionDataset(Dataset):
    def __init__(self, features_files, targets_files, mean, std, drop_feature_columns=None, target_column='token0_drop_10%_15slots'):
        self.features_files = features_files
        self.targets_files = targets_files
        self.mean = mean
        self.std = std
        self.drop_feature_columns = drop_feature_columns if drop_feature_columns is not None else []
        self.target_column = target_column

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        f_file = self.features_files[idx]
        t_file = self.targets_files[idx]
        
        features_df = pl.read_parquet(f_file, use_pyarrow=True)
        targets_df = pl.read_parquet(t_file, use_pyarrow=True)

        # Drop specified columns
        features_df = features_df.drop(self.drop_feature_columns)
        features_df = (features_df - self.mean) / self.std
        features_tensor = torch.tensor(features_df.to_numpy(), dtype=torch.float32)
        targets_tensor = torch.tensor(targets_df[self.target_column].to_numpy(), dtype=torch.float32)
        
        return features_tensor, targets_tensor

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

def compute_scaler_params(features_files, drop_feature_columns=None):
    mean = None
    M2 = None
    n = 0
    drop_feature_columns = drop_feature_columns if drop_feature_columns is not None else []
    
    for f_file in features_files:
        features_df = pl.read_parquet(f_file, use_pyarrow=True)
        features_df = features_df.drop(drop_feature_columns)
        if mean is None:
            mean = features_df.mean(axis=0)
            M2 = ((features_df - mean) ** 2).sum(axis=0)
            n = len(features_df)
        else:
            new_n = len(features_df)
            new_mean = features_df.mean(axis=0)
            new_M2 = ((features_df - new_mean) ** 2).sum(axis=0)
            
            total_n = n + new_n
            delta = new_mean - mean
            mean = (mean * n + new_mean * new_n) / total_n
            M2 += new_M2 + delta ** 2 * n * new_n / total_n
            n = total_n
    
    variance = M2 / (n - 1)
    std = variance ** 0.5
    
    # 保存标准化参数到 JSON 文件
    scaler_params = {
        'mean': mean.tolist(),
        'scale': std.tolist()
    }
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)

    return mean, std

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

def predict(model, data_loader):
    model.eval()  # 设置模型为评估模式
    predictions = []
    with torch.no_grad():
        for features, _ in data_loader:
            outputs = model(features, features)
            preds = torch.sigmoid(outputs).squeeze()
            predictions.extend(preds.cpu().numpy())
    return predictions

def predict_step_by_step(model, features):
    model.eval()  # 设置模型为评估模式
    predictions = []
    with torch.no_grad():
        for i in range(features.size(0)):
            feature = features[i].unsqueeze(0).unsqueeze(0)  # 添加 batch 和 sequence 维度
            preds = model.predict_step(feature).cpu().numpy()
            predictions.append(preds)
    model.reset_memory()  # 清除历史数据
    return np.array(predictions)

if __name__ == "__main__":
    root_dir = 'processed_data'
    drop_feature_columns = ['slot', 'time_of_day_morning', 'time_of_day_afternoon', 'time_of_day_evening', 'time_of_day_night']  # 需要丢弃的特征列
    target_column = 'token0_drop_10%_15slots'
    
    features_files, targets_files = get_all_parquet_files(root_dir)
    
    # 计算标准化参数
    mean, std = compute_scaler_params(features_files, drop_feature_columns)
    
    # 创建 TransactionDataset
    train_dataset = TransactionDataset(features_files, targets_files, mean, std, drop_feature_columns, target_column)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    input_dim = mean.shape[0]  # 假设mean的形状与输入特征维度一致
    d_model = 128
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    # 创建 Transformer 模型
    model = TransformerModel(input_dim=input_dim, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)

    # 计算正负样本权重
    num_pos_samples = 0
    num_neg_samples = 0
    for _, targets in train_loader:
        num_pos_samples += targets.sum().item()
        num_neg_samples += len(targets) - targets.sum().item()
    pos_weight = num_neg_samples / num_pos_samples

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()  # 梯度清零
            outputs = model(features, features)  # 前向传播
            loss = criterion(outputs.squeeze(), targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()  # 累积损失

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        save_model(model, optimizer, epoch, running_loss, path=f'model_epoch_{epoch}.pth')