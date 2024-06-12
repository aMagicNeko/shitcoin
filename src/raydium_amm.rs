use anyhow::Error;
use futures::TryFutureExt;
use solana_sdk::nonce::state::Data;
use solana_sdk::program_error::ProgramError;
use solana_sdk::transaction::Transaction;
use solana_sdk::{pubkey, transaction};
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signer::Signer;
use log::{info, error};
use tokio::task;
use crate::strategy::{self, DataSavingStrategy, Step};
use crate::{subscription::{Instruction, SLOT_BROADCAST_CHANNEL, CURRENT_SLOT}, strategy::Strategy};
use crate::transaction_executor::{gen_associated_token_account, TOKEN_VAULT_MAP, RPC_CLIENT};
use serde::{Deserialize, Serialize};
extern crate base64;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::mpsc::{self, Sender, Receiver};
use lazy_static::lazy_static;
use chrono::Utc;
use raydium_amm::{log::{DepositLog, WithdrawLog, InitLog, SwapBaseInLog, SwapBaseOutLog, LogType}, instruction::{AmmInstruction, swap_base_in}, math::SwapDirection::Coin2PC};
use serum_dex::{instruction::MarketInstruction, state::gen_vault_signer_key};
use anyhow::{anyhow};
//pub const RAY_AMM_ADDRESS: Pubkey = pubkey!("HWy1jotHpo6UqeQxx49dpYYdQB8wj9Qk9MdxwjLvDHB8");
pub const RAY_AMM_ADDRESS: Pubkey = pubkey!("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8");
pub const RAY_LOG_PREFIX: &str = "Program log: ray_log: ";
pub const SOL: Pubkey = pubkey!("So11111111111111111111111111111111111111112");
pub const OPENBOOK_MARKET: Pubkey = pubkey!("srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX");
//pub const OPENBOOK_MARKET: Pubkey = pubkey!("82iPEvGiTceyxYpeLK3DhSwga3R5m4Yfyoydd13CukQ9");
#[derive(Debug)]
pub struct RaydiumAmmPool {
    pub coin_mint: Pubkey,
    pub pc_mint: Pubkey,
    pub ntoken0: u64,
    pub ntoken1: u64,
    pub creator: Pubkey,
    pub creator_token0: Pubkey,
    pub creator_token1: Pubkey,
    pub amm_pool: Pubkey,
    pub amm_authority: Pubkey,
    pub amm_open_orders: Pubkey,
    pub amm_coin_vault: Pubkey,
    pub amm_pc_vault: Pubkey,
    pub market_program: Pubkey,
    pub market: Pubkey,
    pub market_bids: Pubkey,
    pub market_asks: Pubkey,
    pub market_event_queue: Pubkey,
    pub market_coin_vault: Pubkey,
    pub market_pc_vault: Pubkey,
    pub market_vault_signer: Pubkey,
}

impl RaydiumAmmPool {
    pub fn new(coin_mint: Pubkey,
        pc_mint: Pubkey,
        ntoken0: u64,
        ntoken1: u64,
        creator: Pubkey,
        creator_token0: Pubkey,
        creator_token1: Pubkey,
        amm_pool: Pubkey,
        amm_authority: Pubkey,
        amm_open_orders: Pubkey,
        amm_coin_vault: Pubkey,
        amm_pc_vault: Pubkey,
        market_program: Pubkey,
        market: Pubkey,
        market_bids: Pubkey,
        market_asks: Pubkey,
        market_event_queue: Pubkey,
        market_coin_vault: Pubkey,
        market_pc_vault: Pubkey,
        market_vault_signer: Pubkey,) -> Sender<Step> {
        let (tx, mut rx): (Sender<Step>, Receiver<Step>) = mpsc::channel(100);
        task::spawn(async move {
            let mut pool = RaydiumAmmPool {
                coin_mint,
                pc_mint,
                ntoken0,
                ntoken1,
                creator,
                creator_token0,
                creator_token1,
                amm_pool,
                amm_authority,
                amm_open_orders,
                amm_coin_vault,
                amm_pc_vault,
                market_program,
                market,
                market_bids,
                market_asks,
                market_event_queue,
                market_coin_vault,
                market_pc_vault,
                market_vault_signer,
            };
            let sol_as_coin = coin_mint == SOL;
            /*if let Err(e) = pool.execute_swap(80000000, sol_as_coin).await {
                let mut pools = POOL_MANAGER.pools.lock().await;
                pools.remove(&amm_pool);
                error!("pool {} first swap failed:{}", amm_pool, e);
                return
            }*/
            info!("new pool {:?}", pool);
            let mut pending_swap = true;
            let mut strategy = DataSavingStrategy::new(&amm_pool, sol_as_coin);
            let mut slot_rx = SLOT_BROADCAST_CHANNEL.0.subscribe();
            let mut slot: u64 = *CURRENT_SLOT.read().await;
            let mut delta_sol: i64 = 0;
            let mut delta_token: i64 = 0;
            loop {
                tokio::select! {
                    Some(step) = rx.recv() => {
                        pool.ntoken0 = step.token0;
                        pool.ntoken1 = step.token1;
                        /*
                        if step.from == KEYPAIR.pubkey() {
                            pending_swap = false;
                            if sol_as_coin {
                                delta_sol -= step.delta0;
                                delta_token -= step.delta1;
                            }
                            else {
                                delta_sol -= step.delta1;
                                delta_token -= step.delta0;
                            }
                        }
                        else */if strategy.on_transaction(&step) {
                            break;
                        }
                    },
                    Ok(current_slot) = slot_rx.recv() => {
                        slot = current_slot;
                        info!("{}", slot);
                    }
                    else => error!("channel closed"),
                }
            }
            let mut pools = POOL_MANAGER.pools.lock().await;
            pools.remove(&amm_pool);
            info!("pool {} end", amm_pool);
            return
        });
        tx
    }
    /*
    pub async fn execute_swap(&self, amount_in: u64, zero_for_one: bool) -> Result<(), Error> {
        let user_source_owner = KEYPAIR.pubkey();
        let (mint0, mint1) = if zero_for_one {
            (self.coin_mint ,self.pc_mint)
        }
        else {
            (self.pc_mint ,self.coin_mint)
        };
        let user_token_source = {
            let vault_map = TOKEN_VAULT_MAP.read().await;
            match vault_map.get(&mint0) {
                Some(source_account) => *source_account,
                None => {
                    error!("Source associate account not found for {}", self.coin_mint);
                    return Err(anyhow!("Source associate account not found for {}", self.coin_mint));
                }
            }
        };
    
        let (user_token_destination, mut instructions) = {
            let vault_map = TOKEN_VAULT_MAP.read().await;
            match vault_map.get(&mint1) {
                Some(destination_account) => (*destination_account, vec![]),
                None => {
                    drop(vault_map);
                    match gen_associated_token_account(&mint1, &user_source_owner).await {
                        Ok((instr, account)) => (account, instr),
                        Err(e) => return Err(anyhow!("Failed to generate associated token account: {}", e)),
                    }
                }
            }
        };
    
        let instruction = match swap_base_in(
            &RAY_AMM_ADDRESS,
            &self.amm_pool,
            &self.amm_authority,
            &self.amm_open_orders,
            &self.amm_coin_vault,
            &self.amm_pc_vault,
            &self.market_program,
            &self.market,
            &self.market_bids,
            &self.market_asks,
            &self.market_event_queue,
            &self.market_coin_vault,
            &self.market_pc_vault,
            &self.market_vault_signer,
            &user_token_source,
            &user_token_destination,
            &user_source_owner,
            amount_in,
            0,
        ) {
            Ok(instruction) => instruction,
            Err(e) => return Err(anyhow!("Failed to generate raydium swap instruction: {e}")),
        };
    
        instructions.push(instruction);
        let sig = execute_tx_with_comfirm(&instructions).await?;
        // get tx detais here or wait for websocket?
        Ok(())
    }
    */

}

pub struct RaydiumAmmPoolManager {
    pools: Mutex<HashMap<Pubkey, Sender<Step>>>,
}

lazy_static! {
    static ref POOL_MANAGER: Arc<RaydiumAmmPoolManager> = Arc::new(RaydiumAmmPoolManager::new());
}

pub struct OpenbookInfo {
    pub market: Pubkey,
    pub req_queue: Pubkey,
    pub event_queue: Pubkey,
    pub bids: Pubkey,
    pub asks: Pubkey,
    pub coin_vault: Pubkey,
    pub pc_vault: Pubkey,
    pub coin_mint: Pubkey,
    pub pc_mint: Pubkey,
    pub vault_signer: Pubkey
}

lazy_static! {
    pub static ref OPENBOOK_MARKET_CACHE: Mutex<HashMap<Pubkey, OpenbookInfo>> = Mutex::new(HashMap::new());
}

impl RaydiumAmmPoolManager {
    pub fn new() -> Self {
        RaydiumAmmPoolManager {
            pools: Mutex::new(HashMap::new()),
        }
    }

    pub async fn add_pool(&self, instruction: &Instruction, ntoken0: u64, ntoken1: u64) {
        {
            let amm_pool = instruction.accounts[4];
            let amm_authority = instruction.accounts[5];
            let amm_open_orders = instruction.accounts[6];            
            let coin_mint = instruction.accounts[8]; // mint
            let pc_mint = instruction.accounts[9]; // mint
            let amm_coin_vault = instruction.accounts[10];
            let amm_pc_vault = instruction.accounts[11];
            let market_program = instruction.accounts[15];
            let market = instruction.accounts[16];
            // maybe useful
            let creator = instruction.accounts[17];
            let creator_token0 = instruction.accounts[18];
            let creator_token1 = instruction.accounts[19];
            if coin_mint != SOL && pc_mint != SOL {
                return
            }
            let openbook_map = OPENBOOK_MARKET_CACHE.lock().await;
            if let Some(market_info) = openbook_map.get(&market) {
                let sender = RaydiumAmmPool::new(coin_mint,
                    pc_mint,
                    ntoken0,
                    ntoken1,
                    creator,
                    creator_token0,
                    creator_token1,
                    amm_pool,
                    amm_authority,
                    amm_open_orders,
                    amm_coin_vault,
                    amm_pc_vault,
                    market_program,
                    market,
                    market_info.bids,
                    market_info.asks,
                    market_info.event_queue,
                    market_info.coin_vault,
                    market_info.pc_vault,
                    market_info.vault_signer);
                let mut pools = self.pools.lock().await;
                pools.insert(amm_pool.clone(), sender);
            }
            else {
                error!("no openbook info find for {amm_pool}");
            }
        }
    }

    pub async fn on_step(&self, address: &Pubkey, step: Step) {
        let pools = self.pools.lock().await;
        if let Some(sender) = pools.get(address) {
            sender.send(step).await.unwrap();
        }
    }
}


pub async fn decode_ray_log(log: &str, instruction: &Instruction) {
    let bytes = base64::decode_config(log, base64::STANDARD).unwrap();
    //let now = Utc::now();
    //let timestamp = now.timestamp();
    let slot: u64 = *CURRENT_SLOT.read().await;
    match LogType::from_u8(bytes[0]) {
        LogType::Init => {
            let log: InitLog = bincode::deserialize(&bytes).unwrap();
            let ntoken0 = log.coin_amount;
            let ntoken1 = log.pc_amount;
            POOL_MANAGER.add_pool(&instruction, ntoken0, ntoken1).await;
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
                slot
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
                slot
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
            let (delta0, delta1) = if log.direction == Coin2PC as u64{
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
                slot
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
            let (delta0, delta1) = if log.direction == Coin2PC as u64{
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
                slot
            };
            POOL_MANAGER.on_step(&pool_address, step).await;
            info!("swap out log{:?}", log);
        }
    }
}

pub async fn parse_raydium_transaction(instructions: &Vec<Instruction>, logs: &Vec<String>) {
    info!("raydium tx:{:?}:{:?}", instructions, logs);
    let mut j: usize = 0; // instruction iter
    for (i, log) in logs.iter().enumerate() {
        while let Ok(instruction) = AmmInstruction::unpack(&instructions[j].data) {
            match instruction {
                AmmInstruction::PreInitialize(_init_arg) => {
                    j += 1;
                    continue;
                }
                AmmInstruction::Initialize(_init1) => {
                    j += 1;
                    continue;
                }
                AmmInstruction::Initialize2(init2) => {
                    break;
                }
                AmmInstruction::MonitorStep(monitor) => {
                    j += 1;
                    continue;
                }
                AmmInstruction::Deposit(deposit) => {
                    break;
                }
                AmmInstruction::Withdraw(withdraw) => {
                    break;
                }
                AmmInstruction::MigrateToOpenBook => {
                    j += 1;
                    continue;
                }
                AmmInstruction::SetParams(setparams) => {
                    j += 1;
                    continue;
                }
                AmmInstruction::WithdrawPnl => {
                    j += 1;
                    continue;
                }
                AmmInstruction::WithdrawSrm(withdrawsrm) => {
                    j += 1;
                    continue;
                }
                AmmInstruction::SwapBaseIn(swap) => {
                    break;
                }
                AmmInstruction::SwapBaseOut(swap) => {
                    break;
                }
                AmmInstruction::SimulateInfo(simulate) => {
                    j += 1;
                    continue;
                }
                AmmInstruction::AdminCancelOrders(cancel) => {
                    j += 1;
                    continue;
                }
                AmmInstruction::CreateConfigAccount => {
                    j += 1;
                    continue;
                }
                AmmInstruction::UpdateConfigAccount(config_args) => {
                    j += 1;
                    continue;
                }
            }
        }
        decode_ray_log(&log[RAY_LOG_PREFIX.len()..], &instructions[j]).await;
        j += 1;
    }
}

pub async fn parse_serum_instruction(data: &[u8], accounts: &[u8], account_keys: &[Pubkey]) {
    if let Some(instruction) = MarketInstruction::unpack(data) {
        match instruction {
            MarketInstruction::InitializeMarket(ref inner) => {
                let vault_signer_nonce = inner.vault_signer_nonce;
                let market = account_keys[accounts[0] as usize];
                if let Ok(vault_signer) = gen_vault_signer_key(vault_signer_nonce, &market, &OPENBOOK_MARKET) {
                    let open_book = OpenbookInfo{
                        market,
                        req_queue: account_keys[accounts[1] as usize],
                        event_queue: account_keys[accounts[2] as usize],
                        bids: account_keys[accounts[3] as usize],
                        asks: account_keys[accounts[4] as usize],
                        coin_vault: account_keys[accounts[5] as usize],
                        pc_vault: account_keys[accounts[6] as usize],
                        coin_mint: account_keys[accounts[7] as usize],
                        pc_mint: account_keys[accounts[8] as usize],
                        vault_signer
                    };
                    let mut orderbook_map = OPENBOOK_MARKET_CACHE.lock().await;
                    orderbook_map.insert(account_keys[accounts[0] as usize], open_book);
                }
                else {
                    error!("serum gen_vault_signer_key failed for {market}");
                }
            }
            _ => (),
        }
    }
}