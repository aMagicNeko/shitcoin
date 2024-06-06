use solana_sdk::nonce::state::Data;
use solana_sdk::transaction::Transaction;
use solana_sdk::{pubkey, transaction};
use solana_sdk::pubkey::Pubkey;
use log::{info, error};
use tokio::task;
use crate::strategy::{self, DataSavingStrategy, Step};
use crate::{subscription::Instruction, strategy::Strategy};
use serde::{Deserialize, Serialize};
extern crate base64;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::mpsc::{self, Sender, Receiver};
use lazy_static::lazy_static;
use chrono::Utc;

pub const RAY_AMM_ADDRESS: Pubkey = pubkey!("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8");
pub const RAY_LOG_PREFIX: &str = "Program log: ray_log: ";
pub const SOL: Pubkey = pubkey!("So11111111111111111111111111111111111111112");
#[derive(Debug)]
pub struct RaydiumAmmPool {
    pub address: Pubkey,
    pub token0: Pubkey,
    pub token1: Pubkey,
    pub ntoken0: u64,
    pub ntoken1: u64,
    pub creator: Pubkey,
    pub creator_token0: Pubkey,
    pub creator_token1: Pubkey,
}

impl RaydiumAmmPool {
    pub fn new(address: Pubkey, token0: Pubkey, token1: Pubkey, ntoken0: u64, ntoken1: u64, creator: Pubkey, creator_token0: Pubkey, creator_token1: Pubkey) -> Sender<Step> {
        let (tx, mut rx): (Sender<Step>, Receiver<Step>) = mpsc::channel(100);
        task::spawn(async move {
            let pool = RaydiumAmmPool {
                address: address.clone(), // need here
                token0,
                token1,
                ntoken0,
                ntoken1,
                creator,
                creator_token0,
                creator_token1,
            };
            let mut strategy = DataSavingStrategy::new(&address);
            info!("new pool {:?}", pool);
            while let Some(step) = rx.recv().await {
                if strategy.on_transaction(&step, token0 == SOL) {
                    let mut pools = POOL_MANAGER.pools.lock().await;
                    pools.remove(&address);
                    info!("pool {} end", address);
                    return
                }
            }
        });
        tx
    }
}

pub struct RaydiumAmmPoolManager {
    pools: Mutex<HashMap<Pubkey, Sender<Step>>>,
}

lazy_static! {
    static ref POOL_MANAGER: Arc<RaydiumAmmPoolManager> = Arc::new(RaydiumAmmPoolManager::new());
}

impl RaydiumAmmPoolManager {
    pub fn new() -> Self {
        RaydiumAmmPoolManager {
            pools: Mutex::new(HashMap::new()),
        }
    }

    pub async fn add_pool(&self, address: Pubkey, token0: Pubkey, token1: Pubkey, ntoken0: u64, ntoken1: u64, creator: Pubkey, creator_token0: Pubkey, creator_token1: Pubkey) {
        let sender = RaydiumAmmPool::new(address, token0, token1, ntoken0, ntoken1, creator, creator_token0, creator_token1);
        {
            if token0 != SOL && token1 != SOL {
                return
            }
            let mut pools = self.pools.lock().await;
            pools.insert(address.clone(), sender);
        }
    }

    pub async fn on_step(&self, address: &Pubkey, step: Step) {
        let pools = self.pools.lock().await;
        if let Some(sender) = pools.get(address) {
            sender.send(step).await.unwrap();
        }
    }
}

/// LogType enum
#[derive(Debug)]
pub enum LogType {
    Init,
    Deposit,
    Withdraw,
    SwapBaseIn,
    SwapBaseOut,
}

impl LogType {
    pub fn from_u8(log_type: u8) -> Self {
        match log_type {
            0 => LogType::Init,
            1 => LogType::Deposit,
            2 => LogType::Withdraw,
            3 => LogType::SwapBaseIn,
            4 => LogType::SwapBaseOut,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InitLog {
    pub log_type: u8,
    pub time: u64,
    pub pc_decimals: u8,
    pub coin_decimals: u8,
    pub pc_lot_size: u64,
    pub coin_lot_size: u64,
    pub pc_amount: u64,
    pub coin_amount: u64,
    pub market: Pubkey,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DepositLog {
    pub log_type: u8,
    // input
    pub max_coin: u64,
    pub max_pc: u64,
    pub base: u64,
    // pool info
    pub pool_coin: u64,
    pub pool_pc: u64,
    pub pool_lp: u64,
    pub calc_pnl_x: u128,
    pub calc_pnl_y: u128,
    // calc result
    pub deduct_coin: u64,
    pub deduct_pc: u64,
    pub mint_lp: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WithdrawLog {
    pub log_type: u8,
    // input
    pub withdraw_lp: u64,
    // user info
    pub user_lp: u64,
    // pool info
    pub pool_coin: u64,
    pub pool_pc: u64,
    pub pool_lp: u64,
    pub calc_pnl_x: u128,
    pub calc_pnl_y: u128,
    // calc result
    pub out_coin: u64,
    pub out_pc: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SwapBaseInLog {
    pub log_type: u8,
    // input
    pub amount_in: u64,
    pub minimum_out: u64,
    pub direction: u64,
    // user info
    pub user_source: u64,
    // pool info
    pub pool_coin: u64,
    pub pool_pc: u64,
    // calc result
    pub out_amount: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SwapBaseOutLog {
    pub log_type: u8,
    // input
    pub max_in: u64,
    pub amount_out: u64,
    pub direction: u64,
    // user info
    pub user_source: u64,
    // pool info
    pub pool_coin: u64,
    pub pool_pc: u64,
    // calc result
    pub deduct_in: u64,
}

#[repr(u64)]
pub enum SwapDirection {
    PC2Coin = 1u64,
    Coin2Pc = 2u64,
}

pub async fn decode_ray_log(log: &str, instruction: &Instruction) {
    let bytes = base64::decode_config(log, base64::STANDARD).unwrap();
    let now = Utc::now();
    let timestamp = now.timestamp();
    match LogType::from_u8(bytes[0]) {
        LogType::Init => {
            let log: InitLog = bincode::deserialize(&bytes).unwrap();
            let pool_address = instruction.accounts[4];
            let token0 = instruction.accounts[8];
            let token1 = instruction.accounts[9];
            // maybe useful
            let creator = instruction.accounts[17];
            let createor_token0 = instruction.accounts[18];
            let createor_token1 = instruction.accounts[19];
            let ntoken0 = log.coin_amount;
            let ntoken1 = log.pc_amount;
            POOL_MANAGER.add_pool(pool_address, token0, token1, ntoken0, ntoken1, creator, createor_token0, createor_token1).await;
            info!("init log{:?}", log);
        }
        LogType::Deposit => {
            let log: DepositLog = bincode::deserialize(&bytes).unwrap();
            let pool_address = instruction.accounts[1];
            let from = instruction.accounts[12];
            let step = Step {
                from,
                token0: log.pool_coin,
                token1: log.pool_pc,
                delta0: log.deduct_coin as i64,
                delta1: log.deduct_pc as i64,
                timestamp
            };
            POOL_MANAGER.on_step(&pool_address, step).await;
            info!("deposit log{:?}", log);
        }
        LogType::Withdraw => {
            let log: WithdrawLog = bincode::deserialize(&bytes).unwrap();
            const ACCOUNT_LEN: usize = 20;
            let pool_address = instruction.accounts[1];
            let from = if instruction.accounts.len() == ACCOUNT_LEN + 2 || instruction.accounts.len() == ACCOUNT_LEN + 3 {
                instruction.accounts[18]
            }
            else {
                instruction.accounts[16]
            };
            let step = Step {
                from,
                token0: log.pool_coin,
                token1: log.pool_pc,
                delta0: -(log.out_coin as i64),
                delta1: -(log.out_pc as i64),
                timestamp
            };
            POOL_MANAGER.on_step(&pool_address, step).await;
            info!("withdraw log {:?}", log);
        }
        LogType::SwapBaseIn => {
            let log: SwapBaseInLog = bincode::deserialize(&bytes).unwrap();
            let pool_address = instruction.accounts[1];
            const ACCOUNT_LEN: usize = 17;
            let from = if instruction.accounts.len() == ACCOUNT_LEN + 1 {
                instruction.accounts[15]
            }
            else {
                instruction.accounts[14]
            };
            let (delta0, delta1) = if log.direction == SwapDirection::Coin2Pc as u64{
                (log.amount_in as i64, -(log.out_amount as i64))
            }
            else {
                (-(log.out_amount as i64), log.amount_in as i64)
            };
            let step = Step {
                from,
                token0: log.pool_coin,
                token1: log.pool_pc,
                delta0,
                delta1,
                timestamp
            };
            POOL_MANAGER.on_step(&pool_address, step).await;
            info!("swap in log{:?}", log);
        }
        LogType::SwapBaseOut => {
            let log: SwapBaseOutLog = bincode::deserialize(&bytes).unwrap();
            let pool_address = instruction.accounts[1];
            const ACCOUNT_LEN: usize = 17;
            let from = if instruction.accounts.len() == ACCOUNT_LEN + 1 {
                instruction.accounts[15]
            }
            else {
                instruction.accounts[14]
            };
            let (delta0, delta1) = if log.direction == SwapDirection::Coin2Pc as u64{
                (log.deduct_in as i64, -(log.amount_out as i64))
            }
            else {
                (-(log.amount_out as i64), log.deduct_in as i64)
            };
            let step = Step {
                from,
                token0: log.pool_coin,
                token1: log.pool_pc,
                delta0,
                delta1,
                timestamp
            };
            POOL_MANAGER.on_step(&pool_address, step).await;
            info!("swap out log{:?}", log);
        }
    }
}

pub async fn parse_raydium_transaction(instructions: &Vec<Instruction>, logs: &Vec<String>) {
    // TODO: ignore useless instruction
    if instructions.len() != logs.len() {
        error!("ray instructions and logs length not compatible!");
        return;
    }
    info!("raydium tx:{:?}:{:?}", instructions, logs);
    for (i, log) in logs.iter().enumerate() {
        decode_ray_log(&log[RAY_LOG_PREFIX.len()..], &instructions[i]).await;
    }
}

