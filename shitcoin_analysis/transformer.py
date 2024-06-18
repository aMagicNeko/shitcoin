import torch
import torch.nn as nn
import math
from feature_reader import process_files_randomly
from torch.utils.data import Dataset, DataLoader
import os
import polars as pl
import json
import numpy as np
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
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
    def __init__(self, features_list, targets_list):
        self.features_list = features_list
        self.targets_list = targets_list

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        return self.features_list[idx], self.targets_list[idx]

def get_all_parquet_files(root_dir):
    features_files = []
    targets_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('featuresallslots.parquet'):
                features_files.append(os.path.join(root, file))
            elif file.endswith('targetsallslots.parquet'):
                targets_files.append(os.path.join(root, file))
    return features_files, targets_files


def preprocess_data(features_list, targets_list, scaler_path=None, fit_scaler=True):
    processed_features = []
    processed_targets = []
    
    if fit_scaler:
        all_features: pl.DataFrame = pl.concat(features_list)
        mean = all_features.mean(axis=0)
        std = all_features.std(axis=0)
        
        # 保存标准化参数到 JSON 文件
        scaler_params = {
            'mean': mean.tolist(),
            'scale': std.tolist()
        }
        with open(scaler_path, 'w') as f:
            json.dump(scaler_params, f)
    
    else:
        # 从 JSON 文件加载标准化参数
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        mean = np.array(scaler_params['mean'])
        scale = np.array(scaler_params['scale'])

    for features_df, targets_df in zip(features_list, targets_list):
        # 标准化特征
        features_df = (features_df - mean) / scale
        
        # 将 DataFrame 转换为 PyTorch 张量
        features_tensor = torch.tensor(features_df.to_numpy(), dtype=torch.float32)
        targets_tensor = torch.tensor(targets_df['token0_drop_10%_15slots'].to_numpy(), dtype=torch.float32)
        
        processed_features.append(features_tensor)
        processed_targets.append(targets_tensor)
    return processed_features, processed_targets

def load_parquet_files(features_files, targets_files):
    features_list = []
    targets_list = []
    for f_file, t_file in zip(features_files, targets_files):
        features_df = pl.read_parquet(f_file)
        targets_df = pl.read_parquet(t_file)
        features_list.append(features_df)
        targets_list.append(targets_df)
    return features_list, targets_list


if __name__ == "__main___":
    root_dir = 'process_data'
    features_files, targets_files = get_all_parquet_files(root_dir)
    features_list, targets_list = load_parquet_files(features_files, targets_files)
    processed_features, processed_targets = preprocess_data(features_list, targets_list)

    train_dataset = TransactionDataset(processed_features, processed_targets)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 这里batch_size=1是因为每个文件处理为一个序列
    input_dim = processed_features[0].shape[1]
    d_model = 128
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    # 创建 Transformer 模型
    model = TransformerModel(input_dim=input_dim, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)

    num_pos_samples = sum([targets.sum().item() for targets in processed_targets])
    num_neg_samples = sum([len(targets) - targets.sum().item() for targets in processed_targets])
    pos_weight = num_neg_samples / num_pos_samples

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0
        for batch in train_loader:
            # 每个 batch 是一个序列
            for features, targets in batch:
                src = features.unsqueeze(0)  # 添加 batch 维度，使输入形状为 (1, seq_len, input_dim)
                tgt = targets.unsqueeze(0)  # 目标张量也可以使用输入特征，或者使用其他方法生成目标
                optimizer.zero_grad()  # 梯度清零
                outputs = model(src, tgt)  # 前向传播
                targets = tgt.squeeze(0)  # 移除 batch 维度，使目标形状为 (seq_len, input_dim)
                loss = criterion(outputs, targets)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                running_loss += loss.item()  # 累积损失

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")  # 输出平均损失