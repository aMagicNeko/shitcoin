import joblib
from datetime import datetime
import polars as pl
import numpy as np
from backtest import StrategyBase, FeatureExtractor, start_backtest, get_all_files
import json
from util import swap_token_amount_base_in
class RandomForestStrategy(StrategyBase):
    def __init__(self, model_path, std_param_path, prob_thresh, init_in):
        super().__init__()
        self.model = self.load_model(model_path)
        self.position = None
        self.nsol = None
        self.prev_slot = None
        self.prob_thresh = prob_thresh
        with open(std_param_path, 'r') as f:
            self.std_param = json.load(f)
        self.init_in = init_in
    
    def load_model(self, model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model

    def on_slot(self, fe: FeatureExtractor, slot):
        pass_slot = 0
        if self.prev_slot is not None:
            pass_slot = slot - self.prev_slot - 1
        self.prev_slot = slot
        features = fe.compute_features(pass_slot)
        if self.position == None:
            # buy
            self.position = []
            self.nsol = []
            for prob_thresh in self.prob_thresh:
                for init_in in self.init_in:
                    self.position.append(swap_token_amount_base_in(init_in, features['current_token0'], features['current_token1'], True))
                    self.nsol.append(-init_in)
            return
        features_array = np.zeros((len(self.std_param), ))
        i = 0
        for key, value in self.std_param.items():
            features_array[i] = (features[key] - value['mean']) / value['std']
            i += 1
        drop_prob = self.model.predict_proba(features_array)[1]
        i = 0
        for prob_thresh in self.prob_thresh:
            for init_in in self.init_in:
                if drop_prob >= prob_thresh and self.position[i] != 0:
                    self.nsol[i] += swap_token_amount_base_in(self.position[i], features['current_token0'], features['current_token1'], False)
                    self.position[i] = 0
                    print(f"Selling at slot {slot}, profit {self.nsol[i]}, thresh {prob_thresh}")
                i += 1
    
    def on_end(self):
        i = 0
        for prob_thresh in self.prob_thresh:
            for init_in in self.init_in:
                print(f"thresh {prob_thresh} init_in {init_in} profit {self.nsol[i]} left_postion {self.ntoken[i]}")
                i += 1

if __name__ == "__main__":
    init_in = [5000000, 10000000, 20000000, 40000000, 80000000]
    prob_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dir = ''
    files = get_all_files(dir)
    results = None
    for file in files:
        model_path = "random_forest_model.joblib"
        std_param_path = "random_forest_preprocess.json"
        print(f"start test {file}")
        strategy = RandomForestStrategy(model_path, std_param_path, prob_thresh, init_in)
        start_backtest(file, strategy)
        if results is None:
            results = strategy.nsol
        else:
            i = 0
            for prob_thresh in prob_thresh:
                for init_in in init_in:
                    results[i] += strategy.nsol[i]
                    print(f"thresh {prob_thresh} init_in {init_in} accumulate profit {results[i]}")
                    i += 1
