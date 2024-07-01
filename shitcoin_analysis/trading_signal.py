from transformers import AutoformerConfig, AutoformerForPrediction, AutoformerModel
import torch
import os
import polars as pl
import numpy as np
import json
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
class TradingEnv(gym.Env):
    def __init__(self, autoformer_model, device, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.autoformer_model = autoformer_model
        self.device = device
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.current_file = -1
        self.done = False
        self.holdings = 0
        self.current_file_features = None
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(autoformer_model.config.d_model,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.current_file = 0
        self.done = False
        self.holdings = 0
        return self._next_observation()

    def _next_observation(self):
        if self.current_file_features is None or self.current_step >= self.current_file_features.size(1) - prediction_length:
            self.current_file += 1
            self.current_file_features, self.targets = torch.load(batch_files[self.current_file])
            self.current_file_features = self.current_step.to(device)
            self.targets = self.targets.to(device)
            self.current_step = 0
        i = self.current_step
        self.current_step += 1
        available_data_length = min(i+1, context_length)
        if available_data_length > 0:
            past_values[:, :available_data_length] = self.targets[:, i-available_data_length+1:i+1]
            past_time_features[:, :available_data_length] = self.features[:, i-available_data_length+1:i+1, :len(dynamic_feature_col)+len(time_col)]
        future_values = self.targets[:, i+1:i+prediction_length+1]
        future_time_features = self.current_file_features[:, i+1:i+prediction_length+1, :len(dynamic_feature_col)+len(time_col)]
        static_real_features = self.current_file_features[:, 0, len(dynamic_feature_col)+len(time_col):]
        past_values = self.current_file_features[:, :, 0]
        past_time_features = self.current_file_features[:, :, 1:len(dynamic_feature_col)+1]
        static_real_features = self.current_file_features[:, 0, len(dynamic_feature_col)+1:]
        with torch.no_grad():
            autoformer_output = self.autoformer_model(
                past_values=past_values,
                past_time_features=past_time_features,
                static_real_features=static_real_features,
                past_observed_mask=torch.ones(past_values.shape).to(self.device),
                output_hidden_states=True
            ).decoder_hidden_states
        
        return autoformer_output.cpu().numpy()

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0

        if action == 1:  # buy
            if self.balance >= current_price:
                self.balance -= current_price
                self.holdings += 1
        elif action == 2:  # sell
            if self.holdings > 0:
                self.balance += current_price
                self.holdings -= 1
        next_state = self._next_observation()
        if self.current_file == len(batch_files) - 1 and self.current_step >= self.current_file_features.size(1) - prediction_length:
            self.done = True
        profit = self.balance + self.holdings * current_price - self.initial_balance
        reward = profit  # reward is the profit

        return next_state, reward, self.done, {}

    def render(self, mode='human', close=False):
        profit = self.balance + self.holdings * self.data.iloc[self.current_step]['Close'] - self.initial_balance
        print(f'Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holdings}, Profit: {profit}')


batch_files = [os.path.join('padded_batch_data', f) for f in os.listdir('padded_batch_data') if f.endswith('.pt')]
dynamic_feature_col = ['current_token0', 'current_token1', 'cumulative_inflow', 'cumulative_outflow', 'negative_holdings', 'num_addresses', 'max_address_holding', 'top_5_address_holding', 'top_10_address_holding', 'holding_entropy']
# diff it
time_col = ['slot_elapse']
static_feature_col = ['open_token0', 'open_token1']
context_length = 300
lags_length = 7
input_size = 1
prediction_length=5
batch_dir = 'padded_batch_data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载预训练的Autoformer模型
config = AutoformerConfig(
    prediction_length=5,
    context_length=context_length,
    input_size=1,
    lags_sequence=[1, 2, 3, 4, 5, 6, 7],
    num_time_features=len(time_col),
    num_dynamic_real_features=len(dynamic_feature_col),
    num_static_categorical_features=0,
    num_static_real_features=len(static_feature_col),
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

autoformer_model = AutoformerForPrediction(config).to(device)

# 定义交易环境
env = DummyVecEnv([lambda: TradingEnv(batch_files, autoformer_model, device)])

# 创建并训练PPO模型
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_trading_model")

# 加载并使用模型进行预测
model = PPO.load("ppo_trading_model")
obs = env.reset()
for i in range(len(batch_files)):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
