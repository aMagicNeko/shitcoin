from util import *
SELL_MIN_VAL = 1000000 # 0.001 sol
FEE = 300000 # 0.0003 sol
# 新的策略类：价格上涨时卖出
class SellOnRiseStrategy:
    def __init__(self, init_value, up_threshold):
        self.init_value = init_value
        self.nsol = 0
        self.ntoken = 0
        self.my_orders = []
        self.sell = []
        self.max_sol = 0
        self.up_threshold = up_threshold
        self.my_sol = 0
        self.end = False

    def init_process(self, nsol, ntoken):
        self.nsol = nsol
        self.ntoken = ntoken
        self.prev_sol = nsol  # prev tick
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
            print(f"buy {self.init_value} : {self.my_token} : {time} : {self.nsol} : {self.ntoken}")
            self.my_sol = -self.init_value
            self.init_my_token = self.my_token
            self.my_orders.append((time, -self.init_value, self.my_token))
        if self.nsol > self.max_sol:
            self.max_sol = self.nsol
            self.init_my_token = self.my_token  # update here
        if self.prev_sol >= self.nsol or self.my_token == 0:
            self.prev_sol = self.nsol
            return
        ratio_prev, ratio_cur = 1 - self.prev_sol / self.max_sol, 1 - self.nsol / self.max_sol
        #print(f"{ratio_prev}:{ratio_cur}")
        if ratio_cur - ratio_prev >= self.up_threshold:
            sell_ratio = self.up_threshold / (ratio_cur - ratio_prev)
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
            print(f"sell {in_sol}:{-sell_out_token} :{self.my_token} : {time}")
