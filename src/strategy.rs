use solana_sdk::{blake3::Hash, pubkey::Pubkey};
use polars::prelude::*;
use std::{collections::HashMap, f32::consts::E};
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use log::{info, error};
use chrono::prelude::*;
use raydium_amm::math::{Calculator, SwapDirection, U128};
use once_cell::sync::Lazy;
use tokio::sync::RwLock;
use crate::feature_engine::FeatureExtractor;
use std::sync::Arc;
use crate::transaction_executor::KEYPAIR;
use crate::raydium_amm::MY_SOL;
use solana_sdk::signature::Signer;
use anyhow::{anyhow, Error};
use crate::random_forest;
#[derive(Debug, Clone)]
pub struct Step {
    pub from: Pubkey,
    pub token0: f64, // pool before
    pub token1: f64, // pool before
    pub delta0: f64, // pool delta
    pub delta1: f64, // pool delta
    pub slot: u64
}
#[derive(PartialEq)]
pub enum DecisionDirection {
    BUY = 1,
    SELL = 2
}

pub struct SwapDecision {
    pub amount_in: u64,
    pub direction : DecisionDirection,
    pub current_position: u64
}

pub enum OnTransactionRet {
    End(bool),
    SwapDecision(SwapDecision)
}

pub trait Strategy: Send {
    async fn on_transaction(&mut self, step: &Step) -> OnTransactionRet;
    async fn init_process(&mut self) -> Option<SwapDecision>;
    async fn on_slot(&mut self, slot: u64) -> OnTransactionRet;
    async fn on_sell(&mut self);
    async fn update_position(&mut self, position: u64);
}

pub struct DataSavingStrategy {
    file_path: String,
    data: Vec<Step>,
    last_save_time: i64,
}

impl DataSavingStrategy {
    pub fn new(address: &Pubkey) -> Self {
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
        }
    }

    pub fn save_to_parquet(&self) {
        let mut df;
        match DataFrame::new(vec![
            Series::new("From", self.data.iter().map(|step| step.from.to_string()).collect::<Vec<_>>()),
            Series::new("Token0", self.data.iter().map(|step| step.token0).collect::<Vec<_>>()),
            Series::new("Token1", self.data.iter().map(|step| step.token1).collect::<Vec<_>>()),
            Series::new("Delta0", self.data.iter().map(|step| step.delta0).collect::<Vec<_>>()),
            Series::new("Delta1", self.data.iter().map(|step| step.delta1).collect::<Vec<_>>()),
            Series::new("slot", self.data.iter().map(|step| step.slot).collect::<Vec<_>>()),
        ]) {
            Ok(s_df) => df = s_df,
            Err(e) => {
                error!("new polars df error:{:?}", e);
                return
            }
        }
        if let Ok(file) = std::fs::File::create(&self.file_path) {
            if let Err(e) = ParquetWriter::new(file).finish(&mut df) {
                error!("save polars error:{:?}", e);
            }
        }
    }
}

impl Strategy for DataSavingStrategy {
    async fn on_transaction(&mut self, step: &Step) -> OnTransactionRet {
        self.data.push(step.clone());
        info!("Saved transaction: {:?}", step);
        let now = Utc::now();
        let timestamp = now.timestamp();
        if timestamp > self.last_save_time + 600 {
            self.save_to_parquet();
            self.last_save_time = timestamp;
        }
        let nsol = step.token0 + step.delta0;
        // 0.01 sol
        if nsol < 10000000.0 {
            // Save to Parquet file
            self.save_to_parquet();
            OnTransactionRet::End(true)
        }
        else {
            OnTransactionRet::End(false)
        }
    }

    async fn init_process(&mut self) -> Option<SwapDecision> {
        None
    }

    async fn on_slot(&mut self, slot: u64) -> OnTransactionRet {
        OnTransactionRet::End(false)
    }

    async fn on_sell(&mut self) {
    }

    async fn update_position(&mut self, position: u64) {
    }
}

// SnipeStrategy 结构体
pub struct SnipeStrategy {
    position: u64,
    prev_slot: u64,
    fe: FeatureExtractor,
    pending_tx: Option<SwapDecision>
}

impl SnipeStrategy {
    pub fn new(pool_sol: f64, pool_token: f64, slot: u64) -> Self {
        let fe = FeatureExtractor::new(7681, slot, pool_sol as f32, pool_token as f32);
        SnipeStrategy {
            position: 0,
            prev_slot: slot,
            fe,
            pending_tx: None
        }
    }
}

impl Strategy for SnipeStrategy {
    async fn on_transaction(&mut self, step: &Step) -> OnTransactionRet {
        if (step.from == KEYPAIR.pubkey()) {
            // TODO here, give up this, and ensure the safety of query_spl_token
            if step.delta0 < 0.0 {
                self.position = step.delta1 as u64;
                self.pending_tx = None;
            }
        }
        self.fe.update(step);
        OnTransactionRet::End(false)
    }

    async fn init_process(&mut self) -> Option<SwapDecision> {
        self.pending_tx = Some(SwapDecision {
            amount_in: 24000000,
            direction: DecisionDirection::BUY,
            current_position: 0
        });
        Some(SwapDecision {
            amount_in: 24000000,
            direction: DecisionDirection::BUY,
            current_position: 0
        })
    }

    async fn on_slot(&mut self, slot: u64) -> OnTransactionRet {
        if slot <= self.prev_slot {
            return OnTransactionRet::End(false);
        }
        self.prev_slot = slot;
        self.fe.on_slot(slot);
        if let Some(decision) = &self.pending_tx {
            return OnTransactionRet::End(false);
        }
        if self.position == 0 {
            return OnTransactionRet::End(true);
        }
        let features = self.fe.compute_features();
        match random_forest::predict(features) {
            Ok(fall_prob) => {
                info!("{fall_prob}");
                if fall_prob >= 0.08 && self.position > 0 {
                    self.pending_tx = Some(SwapDecision {
                        amount_in: self.position,
                        direction: DecisionDirection::SELL,
                        current_position: self.position
                    });
                    return OnTransactionRet::SwapDecision(SwapDecision {
                        amount_in: self.position,
                        direction: DecisionDirection::SELL,
                        current_position: self.position
                    })
                }
                else {
                    return OnTransactionRet::End(false)
                };
            }
            Err(e) => error!("predict failed: {e}")
        }
        OnTransactionRet::End(false)
    }

    async fn on_sell(&mut self) {
        match &self.pending_tx {
            Some(decision) => {
                match decision.direction {
                    DecisionDirection::SELL => {
                        info!("on sell:{}", decision.amount_in);
                        self.position -= decision.amount_in;
                    }
                    DecisionDirection::BUY => ()
                }
                self.pending_tx = None;
            },
            None => {
                error!("no pending tx exists while on update_position")
            }
        }
    }

    async fn update_position(&mut self, position: u64) {
        info!("update position:{}", position);
        self.position = position;
        self.pending_tx = None;
    }
}
