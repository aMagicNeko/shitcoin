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
        output = self.fc_out(output)  # 平均池化
        #print(f"model out:{output.shape}")
        return torch.sigmoid(output)

class ChunkedTransactionDataset(Dataset):
    def __init__(self, features_files, targets_files, mean, std, chunk_size, drop_feature_columns=None, target_column='token0_drop_10%_15slots'):
        self.features_files = features_files
        self.targets_files = targets_files
        self.mean = mean
        self.std = std
        self.chunk_size = chunk_size
        self.drop_feature_columns = drop_feature_columns if drop_feature_columns is not None else []
        self.target_column = target_column

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        f_file = self.features_files[idx]
        t_file = self.targets_files[idx]
        
        features_df = pl.read_parquet(f_file)
        targets_df = pl.read_parquet(t_file)

        # Drop specified columns
        features_df = features_df.drop(self.drop_feature_columns)
        for i, col in enumerate(features_df.columns):
            features_df = features_df.with_columns((pl.col(col) - self.mean[i]) / self.std[i])
        
        features_tensor = torch.tensor(features_df.to_numpy(), dtype=torch.float32)
        targets_tensor = torch.tensor(targets_df[self.target_column].to_numpy(), dtype=torch.float32).reshape((-1, 1))

        # Split into chunks with overlap to ensure continuity
        num_chunks = (features_tensor.size(0) + self.chunk_size - 1) // self.chunk_size
        features_chunks = [features_tensor[i:i+self.chunk_size] for i in range(0, features_tensor.size(0), self.chunk_size)]
        targets_chunks = [targets_tensor[i:i+self.chunk_size] for i in range(0, targets_tensor.size(0), self.chunk_size)]

        return features_chunks, targets_chunks

def process_in_chunks(model, src_chunks, tgt_chunks, device):
    model.eval()
    outputs = []

    with torch.no_grad():
        memory = None
        for src_chunk, tgt_chunk in zip(src_chunks, tgt_chunks):
            src_chunk = src_chunk.unsqueeze(0).to(device)  # Add batch dimension
            tgt_chunk = tgt_chunk.unsqueeze(0).to(device)  # Add batch dimension
            if memory is not None:
                output = model(src_chunk, tgt_chunk, memory=memory)
            else:
                output = model(src_chunk, tgt_chunk)
            memory = output
            outputs.append(output.cpu())

    return torch.cat(outputs, dim=1)

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
        features_df = pl.read_parquet(f_file)
        features_df = features_df.drop(drop_feature_columns)
        new_M2 = np.zeros((features_df.shape[1]))
        if mean is None:
            mean = features_df.mean(axis=0).to_numpy().reshape((-1,))
            i = 0
            for col in features_df.columns:
                new_M2[i] = ((features_df[col] - mean[i]) ** 2).sum()
                i += 1
            n = len(features_df)
            M2 = new_M2
        else:
            new_n = len(features_df)
            new_mean = features_df.mean(axis=0).to_numpy().reshape((-1,))
            i = 0
            for col in features_df.columns:
                new_M2[i] = ((features_df[col] - mean[i]) ** 2).sum()
                i += 1
            
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

def load_scaler_params(scaler_path='scaler_params.json'):
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    mean = np.array(scaler_params['mean'])
    std = np.array(scaler_params['scale'])
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    root_dir = 'processed_data'
    target_column = 'token0_drop_10%_15slots'
    chunk_size = 100  # 设置分片大小
    drop_feature_columns = ['slot', 'time_of_day']
    features_files, targets_files = get_all_parquet_files(root_dir)

    # 计算标准化参数
    mean, std = load_scaler_params()

    # 创建 ChunkedTransactionDataset
    train_dataset = ChunkedTransactionDataset(features_files, targets_files, mean, std, chunk_size, drop_feature_columns, target_column)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    input_dim = mean.shape[0]
    d_model = 128
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    # 创建 Transformer 模型
    model = TransformerModel(input_dim=input_dim, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
    model = model.to(device)

    # 计算正负样本权重
    num_pos_samples = 0
    num_neg_samples = 0
    for features_chunks, targets_chunks in train_loader:
        for targets in targets_chunks:
            num_pos_samples += targets.sum().item()
            num_neg_samples += len(targets) - targets.sum().item()
    pos_weight = num_neg_samples / num_pos_samples

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features_chunks, targets_chunks in train_loader:
            optimizer.zero_grad()
            outputs = process_in_chunks(model, features_chunks, targets_chunks, device)
            outputs = outputs.squeeze(0)
            targets = torch.cat(targets_chunks).to(device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        save_model(model, optimizer, epoch, running_loss, path=f'model_epoch_{epoch}.pth')
