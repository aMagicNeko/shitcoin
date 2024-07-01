import collections
import numpy as np
from scipy.stats import entropy
from datetime import datetime, timedelta
import polars as pl
import os
class FeatureExtractor:
    def __init__(self, file_creation_time, window_size=7681):
        self.window_size = window_size
        self.token0_values = collections.deque(maxlen=window_size)
        self.total_inflow = collections.deque(maxlen=window_size)
        self.total_outflow = collections.deque(maxlen=window_size)
        self.top5_holdings = collections.deque(maxlen=window_size)
        self.top10_holdings = collections.deque(maxlen=window_size)
        self.max_holdings = collections.deque(maxlen=window_size)
        self.holding_entropies = collections.deque(maxlen=window_size)
        self.num_addresses = collections.deque(maxlen=window_size)
        self.negative_holdings = collections.deque(maxlen=window_size)
        self.address_holdings = collections.defaultdict(float)
        self.current_slot = None
        self.current_inflow = 0
        self.current_outflow = 0
        self.current_token0 = 0
        self.current_token1 = 0
        self.open_token0 = None
        self.open_token1 = None
        self.open_liquidity = None
        self.open_time = None
        self.start_slot = None
        self.inslot_order = {}
        self.file_creation_time = file_creation_time
        self.prev_slot = None

    def on_slot(self, slot):
        if self.current_slot != slot and self.current_slot is not None:
            # Update for the current slot
            slot_inflow = 0
            slot_outflow = 0
            for from_addr, delta in self.inslot_order.items():
                if from_addr not in self.address_holdings:
                    self.address_holdings[from_addr] = 0
                self.address_holdings[from_addr] -= delta[1] # -delta1
                if delta[0] > 0:
                    slot_inflow += delta[0]
                else:
                    slot_outflow += delta[0]
            self.current_inflow += slot_inflow
            self.current_outflow += slot_outflow
            self.inslot_order = {}
            for s in range(self.current_slot, slot):
                self._append_current_slot_data()
        self.prev_slot = self.current_slot
        self.current_slot = slot

    def update(self, slot, token0, token1, delta0, delta1, from_address):
        # Initialize opening values on the first slot
        if self.open_token0 is None:
            self.open_token0 = token0
            self.open_token1 = token1
            self.open_liquidity = token0 * token1
            self.start_slot = slot
            self.prev_slot = slot

        if self.current_slot != slot and self.current_slot is not None:
            # Update for the current slot
            slot_inflow = 0
            slot_outflow = 0
            for from_addr, delta in self.inslot_order.items():
                if from_addr not in self.address_holdings:
                    self.address_holdings[from_addr] = 0
                self.address_holdings[from_addr] -= delta[1] # -delta1
                if delta[0] > 0:
                    slot_inflow += delta[0]
                else:
                    slot_outflow += delta[0]
            self.current_inflow += slot_inflow
            self.current_outflow += slot_outflow
            self.inslot_order = {}
            for s in range(self.current_slot, slot):
                self._append_current_slot_data()
            self.prev_slot = self.current_slot
        self.current_slot = slot
        self.current_token0 = token0 + delta0
        self.current_token1 = token1 + delta1
        # Update address holdings
        if from_address not in self.inslot_order:
            self.inslot_order[from_address] = [0, 0]
        self.inslot_order[from_address][0] += delta0
        self.inslot_order[from_address][1] += delta1

    def _append_current_slot_data(self):
        self.token0_values.append(self.current_token0)
        self.total_inflow.append(self.current_inflow)
        self.total_outflow.append(self.current_outflow)

        positive_holdings = [h for h in self.address_holdings.values() if h > 0]
        negative_holdings = sum([h for h in self.address_holdings.values() if h < 0])
        total_token0 = sum(positive_holdings)

        sorted_holdings = sorted(positive_holdings, reverse=True)
        top5_holding = sum(sorted_holdings[:5]) / total_token0 if total_token0 > 0 else 0
        top10_holding = sum(sorted_holdings[:10]) / total_token0 if total_token0 > 0 else 0
        max_holding = sorted_holdings[0] / total_token0
        self.top5_holdings.append(top5_holding)
        self.top10_holdings.append(top10_holding)
        self.max_holdings.append(max_holding)
        
        holding_distribution = np.array(positive_holdings) / total_token0 if total_token0 > 0 else np.array([0])
        self.holding_entropies.append(entropy(holding_distribution))

        self.num_addresses.append(len(positive_holdings))
        self.negative_holdings.append(negative_holdings)

    def compute_features(self, past_slot=0):
        # past_slot: skip num of slots
        if past_slot >= 7680:
            past_slot = 7680
        features = {}
        features['slot'] = self.prev_slot
        features['open_token0'] = self.open_token0
        features['open_token1'] = self.open_token1
        features['open_liquidity'] = self.open_liquidity
        features['cumulative_inflow'] = self.current_inflow
        features['cumulative_outflow'] = self.current_outflow
        features['current_token0'] = self.current_token0
        features['current_token1'] = self.current_token1
        current_liquidity = self.current_token0 * self.current_token1
        features['current_liquidity_ratio'] = current_liquidity / self.open_liquidity if self.open_liquidity else 0
        features['current_to_open_token0_ratio'] = self.current_token0 / self.open_token0 if self.open_token0 else 0
        features['slot_elapse'] = self.prev_slot - self.start_slot

        slot_windows = [15, 30, 60, 120, 240, 480, 960, 1920, 3840, 7680]
        for window in slot_windows:
            window_token0 = self.token0_values[-window-past_slot-1] if len(self.token0_values) > window+past_slot else self.open_token0
            features[f'token0_value_{window}slots'] = window_token0
            features[f'token0_relative_value_{window}slots'] = window_token0 / self.open_token0 if self.open_token0 != 0 else 0
            features[f'token0_diff_value_{window}slots'] = self.token0_values[-1] - window_token0
            features[f'token0_relative_diff_value_{window}slots'] = (self.token0_values[-1] - window_token0) / self.open_token0 if self.open_token0 != 0 else 0

            window_inflow = self.total_inflow[-window-past_slot-1] if len(self.total_inflow) > window+past_slot else 0
            window_outflow = self.total_outflow[-window-past_slot-1] if len(self.total_outflow) > window+past_slot else 0
            features[f'inflow_{window}slots'] = self.total_inflow[-1] - window_inflow
            features[f'outflow_{window}slots'] = self.total_outflow[-1] - window_outflow
        # flow diff
        for window in slot_windows[:-1]:
            features[f'inflow_diff_{window}slots'] = 2 * features[f'inflow_{window}slots'] - features[f'inflow_{2 * window}slots']
            features[f'outflow_diff_{window}slots'] = 2 * features[f'outflow_{window}slots'] - features[f'outflow_{2 * window}slots']
        positive_holdings = [h for h in self.address_holdings.values() if h > 0]
        total_token0 = sum(positive_holdings)
        address_proportions = [v / total_token0 for v in positive_holdings]

        features['negative_holdings'] = sum(h for h in self.address_holdings.values() if h < 0)
        features['num_addresses'] = len(positive_holdings)
        features['max_address_holding'] = max(positive_holdings) / total_token0 if total_token0 > 0 else 0
        features['top_5_address_holding'] = sum(sorted(positive_holdings, reverse=True)[:5]) / total_token0 if total_token0 > 0 else 0
        features['top_10_address_holding'] = sum(sorted(positive_holdings, reverse=True)[:10]) / total_token0 if total_token0 > 0 else 0
        features['holding_entropy'] = entropy(address_proportions)

        for window in slot_windows:
            window_top5_holdings = self.top5_holdings[-window-past_slot-1] if len(self.top5_holdings) > window+past_slot else 1
            features[f'top_5_address_holding_diff_{window}slots'] = self.top5_holdings[-1] - window_top5_holdings

            window_top10_holdings = self.top10_holdings[-window-past_slot-1] if len(self.top10_holdings) > window+past_slot else 1
            features[f'top_10_address_holding_diff_{window}slots'] = self.top10_holdings[-1] - window_top10_holdings

            window_max_holdings = self.max_holdings[-window-past_slot-1] if len(self.max_holdings) > window+past_slot else 1
            features[f'max_address_holding_diff_{window}slots'] = self.max_holdings[-1] - window_max_holdings
            
            window_entropy = self.holding_entropies[-window-past_slot-1] if len(self.holding_entropies) > window+past_slot else 0
            features[f'holding_entropy_diff_{window}slots'] = self.holding_entropies[-1] - window_entropy

            window_num_addresses = self.num_addresses[-window-past_slot-1] if len(self.num_addresses) > window+past_slot else 0
            features[f'num_addresses_diff_{window}slots'] = self.num_addresses[-1] - window_num_addresses

            window_negative_holdings = self.negative_holdings[-window-past_slot-1] if len(self.negative_holdings) > window+past_slot else 0
            features[f'negative_holdings_diff_{window}slots'] = self.negative_holdings[-1] - window_negative_holdings

        current_time = self.file_creation_time + timedelta(milliseconds=int(features['slot_elapse'] * 400))
        features['time_of_day_morning'] = 1 if 6 <= current_time.hour < 12 else 0
        features['time_of_day_afternoon'] = 1 if 12 <= current_time.hour < 18 else 0
        features['time_of_day_evening'] = 1 if 18 <= current_time.hour < 24 else 0
        features['time_of_day_night'] = 1 if current_time.hour < 6 or current_time.hour >= 24 else 0

        return features

class StrategyBase:
    # template of a strategy class
    def __init__(self):
        self.ntoken = None
        self.nsol = None
        self.prev_slot = None
    
    def on_slot(self, fe: FeatureExtractor, slot):
        pass
    
    def on_end(self):
        pass

def start_backtest(file_path, strategy: StrategyBase):
    file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
    datas = pl.read_parquet(file_path)
    fe = FeatureExtractor(file_creation_time)
    prev_slot = None
    for data in datas.rows(named=True):
        if data['slot'] != prev_slot and prev_slot is not None:
            fe.on_slot(data['slot'])
            strategy.on_slot(fe, data['slot'])
            #fe.compute_features(data['slot'] - prev_slot - 1)
        prev_slot = data['slot']
        fe.update(data['slot'], data['Token0'], data['Token1'], data['Delta0'], data['Delta1'], data['From'])
    fe.on_slot(datas[-1, 'slot'] + 1)
    #fe.compute_features()
    strategy.on_slot(fe, datas[-1, 'slot'] + 1)
    strategy.on_end()

def get_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths
