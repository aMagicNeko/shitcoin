from util import *
from backtest import *
import logging

SELL_MIN_VAL = 1000000 # 0.001 sol
FEE = 300000 # 0.0003 sol
class SellOnFallStrategy:
    def __init__(self, init_value, sell_map):
        self.init_value = init_value
        self.nsol = 0
        self.ntoken = 0
        self.my_orders = []
        self.sell = []
        self.max_sol = 0
        self.sell_map = sell_map
        self.my_sol = 0
        self.end = False
    def init_process(self, nsol, ntoken):
        self.nsol = nsol
        self.ntoken = ntoken
        self.prev_sol = nsol # prev tick
        self.my_token = 0
        self.max_sol = self.nsol

    def on_price(self, token0, token1):
        self.nsol = token0
        self.ntoken = token1

    def on_tick(self, time):
        if self.end:
            return
        if self.my_token == 0: 
            # the first buy
            self.my_token += swap_token_amount_base_in(self.init_value, self.nsol, self.ntoken, True)
            logging.info(f"buy {self.init_value} : {self.my_token} : {time} : {self.nsol} : {self.ntoken}")
            if self.my_token == 0:
                self.end = True
            self.my_sol = -self.init_value
            self.init_my_token = self.my_token
            self.my_orders.append((time, -self.init_value, self.my_token))
        if self.nsol > self.max_sol:
            self.max_sol = self.nsol
            self.init_my_token = self.my_token #update here
        if self.prev_sol <= self.nsol or self.my_token == 0:
            self.prev_sol = self.nsol
            return
        ratio_prev, ratio_cur = 1 - self.prev_sol / self.max_sol, 1 - self.nsol / self.max_sol
        #print(f"{ratio_prev}:{ratio_cur}")
        for i in range(len(self.sell_map)):
            if ratio_prev < self.sell_map[i][0]:
                prev_idx = i
                break
        cur_idx = len(self.sell_map)
        for i in range(len(self.sell_map)):
            if ratio_cur < self.sell_map[i][0]:
                cur_idx = i
                break
        if prev_idx > cur_idx:
            return
        sell_ratio = 0
        for i in range(prev_idx, cur_idx):
            sell_ratio += self.sell_map[i][1]
        sell_out_token = int(sell_ratio * self.init_my_token)
        in_sol = swap_token_amount_base_in(sell_out_token, self.nsol, self.ntoken, False)
        if in_sol < SELL_MIN_VAL:
            sell_out_token = self.my_token
            in_sol = swap_token_amount_base_in(sell_out_token, self.nsol, self.ntoken, False)
        self.my_sol += in_sol - FEE
        self.my_token -= sell_out_token
        if self.my_token <= 10:
            self.end = True
        self.my_orders.append((time, in_sol, -sell_out_token))
        logging.info(f"sell {in_sol}:{-sell_out_token} :{self.my_token} : {time}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="fall_logs.txt", format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',datefmt='%d-%b-%y %I:%M:%S %p')
    files = os.listdir(folder_path)
    result = {}
    init_val = [5000000, 10000000, 20000000, 40000000, 80000000]
    sell_maps = [[(0.1, 1)], 
                 [(0.2, 1)], 
                 [(0.3, 1)], 
                 [(0.4, 1)], 
                 [(0.5, 1)], 
                 [(0.6, 1)], 
                 [(0.1, 0.5), (0.2, 0.5)], 
                 [(0.1, 0.3), (0.2, 0.3), (0.3, 0.4)], 
                 [(0.1, 0.2), (0.2, 0.2), (0.3, 0.2), (0.4, 0.2), (0.5, 0.2)]]
    for i in range(len(init_val)):
        for j in range(len(sell_maps)):
            result[(i, j)] = []
    for filename in files:
        try:
            df = read(filename)
            df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')  # 确保时间戳正确
            df = df.dropna(subset=['time'])  # 移除无效时间戳
            for i in range(len(init_val)):
                for j in range(len(sell_maps)):
                    #data, filter_data = process_transaction_data(df)
                    strategy = SellOnFallStrategy(init_val[i], sell_maps[j]) # 0.02sol
                    out_sol, strategy_orders = backtest(strategy, df)
                    #plot_inflow_outflow(data, filter_data, "Inflow and Outflow with Strategy Orders", strategy_orders)
                    result[(i, j)].append((filename, strategy.my_sol))
                    logging.info(f"strategy end {i}, {j}, {filename}: {strategy.my_sol}")
                    #orders.append((filename, strategy.my_orders))
        except:
            continue
    for i in range(len(init_val)):
        for j in range(len(sell_maps)):
            sum = 0
            for x in result[(i, j)]:
                sum += x[1]
            logging.info(f"{i}, {j}: {sum}")