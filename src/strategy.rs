use solana_sdk::pubkey::Pubkey;
use xlsxwriter::*;
use std::fs::File;
use std::path::Path;
use std::sync::Mutex;
use log::info;
#[derive(Debug, Clone)]
pub struct Step {
    pub from: Pubkey,
    //pub to: Pubkey,  maybe different from `from' some time, but that's the rare case
    pub token0: u64, // pool before
    pub token1: u64, // pool before
    pub delta0: i64, // pool delta
    pub delta1: i64, // pool delta
    pub timestamp: i64
}

pub trait Strategy: Send {
    fn on_transaction(&mut self, step: &Step, token0_sol: bool) -> bool;
}

pub struct DataSavingStrategy {
    file_path: String,
    data: Vec<Step>,
    last_save_time: i64
}

impl DataSavingStrategy {
    pub fn new(address: &Pubkey) -> Self {
        let file_path = format!("{}_transactions.xlsx", address.to_string());
        DataSavingStrategy {
            file_path: file_path,
            data: Vec::new(),
            last_save_time: 0
        }
    }

    pub fn save_to_excel(&self) {
        let workbook = Workbook::new(&self.file_path).unwrap();
        let mut sheet = workbook.add_worksheet(None).unwrap();

        // Write header
        sheet.write_string(0, 0, "From", None).unwrap();
        sheet.write_string(0, 1, "Token0", None).unwrap();
        sheet.write_string(0, 2, "Token1", None).unwrap();
        sheet.write_string(0, 3, "Delta0", None).unwrap();
        sheet.write_string(0, 4, "Delta1", None).unwrap();
        sheet.write_string(0, 5, "time", None).unwrap();
        for (i, step) in self.data.iter().enumerate() {
            let row = (i + 1) as u32;
            sheet.write_string(row, 0, &step.from.to_string(), None).unwrap();
            sheet.write_number(row, 1, step.token0 as f64, None).unwrap();
            sheet.write_number(row, 2, step.token1 as f64, None).unwrap();
            sheet.write_number(row, 3, step.delta0 as f64, None).unwrap();
            sheet.write_number(row, 4, step.delta1 as f64, None).unwrap();
            sheet.write_number(row, 5, step.timestamp as f64, None).unwrap();
        }

        workbook.close().unwrap();
        let after_path = format!("coin_data/{}", self.file_path);
        std::fs::rename(&self.file_path, after_path).unwrap();
    }
}

impl Strategy for DataSavingStrategy {
    fn on_transaction(&mut self, step: &Step, token0_sol: bool) -> bool {
        self.data.push(step.clone());
        info!("Saved transaction: {:?}", step);
        
        if step.timestamp > self.last_save_time + 600 {
            self.save_to_excel();
            self.last_save_time = step.timestamp;
        }
        let nsol = if token0_sol {
            step.token0 as i64 + step.delta0
        }
        else {
            step.token1 as i64 + step.delta1
        };
        // 0.01 sol
        if nsol < 10000000 {
            // Save to Excel file
            self.save_to_excel();
            true
        }
        else {
            false
        }
    }
}
