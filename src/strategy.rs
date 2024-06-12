use solana_sdk::pubkey::Pubkey;
use polars::prelude::*;
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use log::info;
use chrono::prelude::*;

#[derive(Debug, Clone)]
pub struct Step {
    pub from: Pubkey,
    pub token0: u64, // pool before
    pub token1: u64, // pool before
    pub delta0: i64, // pool delta
    pub delta1: i64, // pool delta
    pub slot: u64
}

pub trait Strategy: Send {
    fn on_transaction(&mut self, step: &Step) -> bool;
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
    fn on_transaction(&mut self, step: &Step) -> bool {
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
            true
        }
        else {
            false
        }
    }
}
