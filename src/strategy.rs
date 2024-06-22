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
use ort::{GraphOptimizationLevel, Session, DynValue, Map, Allocator, ValueRef, DynMap, MapValueType, DynSequenceValueType, DynValueTypeMarker};
use ndarray::{ArrayD, IxDyn, Array1, Array2};
use crate::transaction_executor::KEYPAIR;
use crate::raydium_amm::MY_SOL;
use solana_sdk::signature::Signer;
use anyhow::{anyhow, Error};
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
    pub direction : DecisionDirection
}

pub enum OnTransactionRet {
    End(bool),
    SwapDecision(SwapDecision)
}

pub trait Strategy: Send {
    async fn on_transaction(&mut self, step: &Step) -> OnTransactionRet;
    async fn init_process(&mut self) -> Option<SwapDecision>;
    async fn on_slot(&mut self, slot: u64) -> OnTransactionRet;
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
}

static RANDOM_FOREST: Lazy<Session> = Lazy::new(|| {
    Session::builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .unwrap()
                .with_intra_threads(4)
                .unwrap()
                .commit_from_file("random_forest_model.onnx")
                .unwrap()
});

pub fn print_model_dim() {
    info!("random forest model input:{:?}, output{:?}", RANDOM_FOREST.inputs, RANDOM_FOREST.outputs);
}

pub fn predict(features: Vec<f32>) -> Result<f32, Error> {
    let input_data = Array2::from_shape_vec((1, 155), features).expect("Failed to create input array");

    let input_tensor = ort::inputs![input_data].expect("Failed to create input tensor");

    let outputs = RANDOM_FOREST.run(input_tensor).expect("Failed to run prediction");
    info!("predict: {:?}", outputs);
    let allocator = Allocator::default();
    let predictions_ref = outputs["output_probability"].try_extract_sequence::<DynValueTypeMarker>(&allocator)?;

    for pred in predictions_ref {
        let map: HashMap<i64, f32> = pred.try_extract_map(&allocator)?;
        let probability = map.get(&1).expect("map error");
        return Ok(*probability);
    }
    Err(anyhow!("no resut"))
}

// SnipeStrategy 结构体
pub struct SnipeStrategy {
    delta_sol: f64,
    delta_token: f64,
    prev_slot: u64,
    fe: FeatureExtractor,
}

impl SnipeStrategy {
    pub fn new(pool_sol: f64, pool_token: f64, slot: u64) -> Self {
        let fe = FeatureExtractor::new(7681, slot, pool_sol as f32, pool_token as f32);
        SnipeStrategy {
            delta_sol: 0.0,
            delta_token: 0.0,
            prev_slot: slot,
            fe,
        }
    }
}

impl Strategy for SnipeStrategy {
    async fn on_transaction(&mut self, step: &Step) -> OnTransactionRet {
        self.fe.update(step);
        if step.from == KEYPAIR.pubkey() {
            let mut my_sol = MY_SOL.write().await;
            *my_sol -= step.delta0;
            self.delta_token -= step.delta1;
            self.delta_sol -= step.delta0;
            info!("my order:{:?}", step);
            if self.delta_token < 1.0 {
                return OnTransactionRet::End(true);
            }
        }
        OnTransactionRet::End(false)
    }

    async fn init_process(&mut self) -> Option<SwapDecision> {
        Some(SwapDecision {
            amount_in: 8000000,
            direction: DecisionDirection::BUY,
        })
    }

    async fn on_slot(&mut self, slot: u64) -> OnTransactionRet {
        if slot <= self.prev_slot {
            return OnTransactionRet::End(false);
        }
        self.prev_slot = slot;
        self.fe.on_slot(slot);
        let features = self.fe.compute_features();
        match predict(features) {
            Ok(fall_prob) => {
                info!("{fall_prob}");
                if fall_prob >= 0.13 && self.delta_token > 0.001 {
                    return OnTransactionRet::SwapDecision(SwapDecision {
                        amount_in: self.delta_token as u64,
                        direction: DecisionDirection::SELL
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
}
