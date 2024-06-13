use solana_sdk::pubkey::Pubkey;
use polars::prelude::*;
use std::f32::consts::E;
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use log::info;
use chrono::prelude::*;
use raydium_amm::math::{Calculator, SwapDirection};
use once_cell::sync::Lazy;
use tokio::sync::RwLock;
#[derive(Debug, Clone)]
pub struct Step {
    pub from: Pubkey,
    pub token0: u64, // pool before
    pub token1: u64, // pool before
    pub delta0: i64, // pool delta
    pub delta1: i64, // pool delta
    pub slot: u64
}

pub struct SwapDecision {
    pub amount_in: u64,
    pub zero_for_one: bool
}

pub enum OnTransactionRet {
    End(bool),
    SwapDecision(SwapDecision)
}

pub trait Strategy: Send {
    fn on_transaction(&mut self, step: &Step) -> OnTransactionRet;
    fn on_buy(&mut self, step: &Step);
    fn on_sell(&mut self, step:& Step) -> bool;
    fn init_process(&mut self) -> Option<SwapDecision>;
}

pub struct DataSavingStrategy {
    file_path: String,
    data: Vec<Step>,
    last_save_time: i64,
    token0_sol: bool
}

impl DataSavingStrategy {
    pub fn new(address: &Pubkey, token0_sol: bool) -> Self {
        let current_date = Utc::now().format("%Y-%m-%d").to_string();
        let folder_path = format!("coin_data/{}", current_date);
        if !Path::new(&folder_path).exists() {
            fs::create_dir_all(&folder_path).unwrap();
        }
        let file_path = format!("{}/{}.parquet", folder_path, address.to_string());
        DataSavingStrategy {
            file_path: file_path,
            data: Vec::new(),
            last_save_time: 0,
            token0_sol
        }
    }

    pub fn save_to_parquet(&self) {
        let mut df;
        if self.token0_sol {
            df = DataFrame::new(vec![
                Series::new("From", self.data.iter().map(|step| step.from.to_string()).collect::<Vec<_>>()),
                Series::new("Token0", self.data.iter().map(|step| step.token0).collect::<Vec<_>>()),
                Series::new("Token1", self.data.iter().map(|step| step.token1).collect::<Vec<_>>()),
                Series::new("Delta0", self.data.iter().map(|step| step.delta0).collect::<Vec<_>>()),
                Series::new("Delta1", self.data.iter().map(|step| step.delta1).collect::<Vec<_>>()),
                Series::new("slot", self.data.iter().map(|step| step.slot).collect::<Vec<_>>()),
            ]).unwrap();
        }
        else {
            df = DataFrame::new(vec![
                Series::new("From", self.data.iter().map(|step| step.from.to_string()).collect::<Vec<_>>()),
                Series::new("Token0", self.data.iter().map(|step| step.token1).collect::<Vec<_>>()),
                Series::new("Token1", self.data.iter().map(|step| step.token0).collect::<Vec<_>>()),
                Series::new("Delta0", self.data.iter().map(|step| step.delta1).collect::<Vec<_>>()),
                Series::new("Delta1", self.data.iter().map(|step| step.delta0).collect::<Vec<_>>()),
                Series::new("slot", self.data.iter().map(|step| step.slot).collect::<Vec<_>>()),
            ]).unwrap();
        }
        let file = std::fs::File::create(&self.file_path).expect("Could not create file.");
        ParquetWriter::new(file).finish(&mut df).expect("Could not write file.");
    }
}

impl Strategy for DataSavingStrategy {
    fn on_transaction(&mut self, step: &Step) -> OnTransactionRet {
        self.data.push(step.clone());
        info!("Saved transaction: {:?}", step);
        let now = Utc::now();
        let timestamp = now.timestamp();
        if timestamp > self.last_save_time + 600 {
            self.save_to_parquet();
            self.last_save_time = timestamp;
        }
        let nsol = if self.token0_sol {
            step.token0 as i64 + step.delta0
        }
        else {
            step.token1 as i64 + step.delta1
        };
        // 0.01 sol
        if nsol < 10000000 {
            // Save to Parquet file
            self.save_to_parquet();
            OnTransactionRet::End(true)
        }
        else {
            OnTransactionRet::End(false)
        }
    }

    fn on_buy(&mut self, step: &Step) {

    }

    fn on_sell(&mut self, step:& Step) -> bool {
        false
    }

    fn init_process(&mut self) -> Option<SwapDecision> {
        None
    }

}

pub struct FallSellStrategy {
    init_my_sol_in: u64,
    highest_sol: u64,
    //prev_sol: u64,
    sell_map: Vec<(f64, f64)>,
    token0_sol: bool,
    delta_sol: i64,
    delta_token: i64,
    prev_slot: u64
}

const MIN_SOL_VAL: u64 = 10000; // 0.00001 sol
const SELL_MIN_SOL: u64 = 1000000; // 0.001 sol
impl FallSellStrategy {
    pub fn new(init_sol: u64, token0_sol: bool) -> Self {
        //let sell_map = vec![(0.1, 0.2), (0.2, 0.2), (0.3, 0.2), (0.4, 0.2), (0.5, 0.2)];
        let sell_map = vec![(0.1, 0.5), (0.2, 0.5)];
        FallSellStrategy {
            init_my_sol_in: 0,
            highest_sol: init_sol,
            //prev_sol: init_sol,
            sell_map,
            token0_sol,
            delta_sol: 0,
            delta_token: 0,
            prev_slot: 0
        }
    }
}
impl Strategy for FallSellStrategy {
    fn on_buy(&mut self, step: &Step) {
        let cur_sol = if self.token0_sol {
            self.delta_sol -= step.delta0;
            self.delta_token -= step.delta1;
            step.token0 as i64 + step.delta0
        }
        else {
            self.delta_sol -= step.delta1;
            self.delta_token -= step.delta0;
            step.token1 as i64 + step.delta1
        };
        self.init_my_sol_in = self.delta_sol as u64;
        self.highest_sol = cur_sol as u64; // prevent immediate sell
        info!("on buy: {}: {}", self.delta_sol, self.delta_token);
    }

    fn on_sell(&mut self, step:& Step) -> bool {
        if self.token0_sol {
            self.delta_sol -= step.delta0;
            self.delta_token -= step.delta1;
        }
        else {
            self.delta_sol -= step.delta1;
            self.delta_token -= step.delta0;
        }
        info!("on sell: {}: {}", self.delta_sol, self.delta_token);
        self.delta_token > 0
    }

    fn on_transaction(&mut self, step: &Step) -> OnTransactionRet {
        let (cur_sol, cur_token, prev_sol) = if self.token0_sol {
            (step.token0 as i64 + step.delta0, step.token1 as i64 + step.delta1, step.token0)
        }
        else {
            (step.token1 as i64 + step.delta1, step.token0 as i64 + step.delta0, step.token1)
        };
        if step.slot != self.prev_slot {
            self.prev_slot = step.slot;
            if prev_sol > self.highest_sol {
                self.highest_sol = prev_sol;
            }
        }
        let mut total_sell_ratio = 0.0;
        if cur_sol < prev_sol as i64 {
            let prev_ratio = if prev_sol < self.highest_sol {
                (self.highest_sol - prev_sol) as f64 / self.highest_sol as f64
            }
            else {
                0.0
            };
            let cur_ratio = if cur_sol < self.highest_sol as i64 {
                (self.highest_sol as i64 - cur_sol) as f64 / self.highest_sol as f64
            }
            else {
                0.0
            };
            let mut prev_idx = 0;
            for (i, (fall_ratio, sell_ratio)) in self.sell_map.iter().enumerate() {
                if *fall_ratio > prev_ratio {
                    prev_idx = i;
                    break;
                }
            }
            for (i, (fall_ratio, sell_ratio)) in self.sell_map.iter().enumerate() {
                if *fall_ratio > cur_ratio {
                    break;
                }
                if i >= prev_idx {
                    total_sell_ratio += sell_ratio;
                }
            }
        }
        info!("on transaction {:?} {:?}, max_sol:{}, sell_ratio:{}, down_ratio:{}, delta_sol:{}, delta_token:{}", step, Utc::now(), self.highest_sol, total_sell_ratio, 1.0 - cur_sol as f64 / self.highest_sol as f64, self.delta_sol, self.delta_token);
        if total_sell_ratio > 0.001 && self.delta_token > 0{
            let mut token_in = (total_sell_ratio * self.delta_token as f64) as u64;
            let sol_out = Calculator::swap_token_amount_base_in(token_in.into(), cur_token.into(), cur_sol.into(), SwapDirection::PC2Coin);
            if sol_out < SELL_MIN_SOL.into() {
                token_in = self.delta_token as u64;
            }
            OnTransactionRet::SwapDecision(SwapDecision{amount_in: token_in, zero_for_one: !self.token0_sol})
        }
        else {
            OnTransactionRet::End(self.delta_token <= 0 && self.init_my_sol_in != 0)
        }
    }

    fn init_process(&mut self) -> Option<SwapDecision> {
        Some(SwapDecision {amount_in: 1000000, zero_for_one: self.token0_sol})
        /*
        self.delta_sol = -1000000;
        self.delta_token = 1232312312312312;
        None
        */
    }
}