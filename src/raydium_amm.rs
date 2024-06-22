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
use crate::strategy::{self, DataSavingStrategy, Step, DecisionDirection, SnipeStrategy};
use crate::{subscription::{Instruction, SLOT_BROADCAST_CHANNEL, CURRENT_SLOT}, strategy::{Strategy, OnTransactionRet}};
use crate::transaction_executor::{gen_associated_token_account, TOKEN_VAULT_MAP, RPC_CLIENT, KEYPAIR, execute_tx_with_comfirm};
use serde::{Deserialize, Serialize};
extern crate base64;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::mpsc::{self, Sender, Receiver};
use lazy_static::lazy_static;
use chrono::Utc;
use raydium_amm::{log::{DepositLog, WithdrawLog, InitLog, SwapBaseInLog, SwapBaseOutLog, LogType}, instruction::{AmmInstruction, swap_base_in}, math::SwapDirection::Coin2PC};
use serum_dex::{instruction::MarketInstruction, state::gen_vault_signer_key};
use anyhow::{anyhow};
use once_cell::sync::Lazy;
use tokio::sync::RwLock;
use spl_token::instruction::{TokenInstruction, AuthorityType};
use solana_program::program_option::COption;
//pub const RAY_AMM_ADDRESS: Pubkey = pubkey!("HWy1jotHpo6UqeQxx49dpYYdQB8wj9Qk9MdxwjLvDHB8");
pub const RAY_AMM_ADDRESS: Pubkey = pubkey!("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8");
pub const RAY_LOG_PREFIX: &str = "Program log: ray_log: ";
pub const SOL: Pubkey = pubkey!("So11111111111111111111111111111111111111112");
pub const OPENBOOK_MARKET: Pubkey = pubkey!("srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX");
//pub const OPENBOOK_MARKET: Pubkey = pubkey!("82iPEvGiTceyxYpeLK3DhSwga3R5m4Yfyoydd13CukQ9");
pub static MY_SOL: Lazy<Arc<RwLock<f64>>> = Lazy::new(|| {
    Arc::new(RwLock::new(0.0))
});

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
    pub amm_target_orders: Pubkey,
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

lazy_static! {
    static ref ONLY_ONE_POOL: Arc<Mutex<i32>> = Arc::new(Mutex::new(0));
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
        amm_target_orders: Pubkey,
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
                amm_target_orders,
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
            /*
            {
                let mut p = ONLY_ONE_POOL.lock().await;
                if *p != 0 {
                    let mut pools = POOL_MANAGER.pools.lock().await;
                        pools.remove(&amm_pool);
                        return
                }
                *p = 1;
            }
            */
            let sol_as_coin = coin_mint == SOL;
            info!("new pool {:?}", pool);
            let mut slot_rx = SLOT_BROADCAST_CHANNEL.0.subscribe();
            let mut slot: u64 = *CURRENT_SLOT.read().await;
            let mut failed_cnt = 0;
            let (buy_direction, sell_direction) = if sol_as_coin {
                (true, false)
            }
            else {
                (false, true)
            };
            //let mut strategy = DataSavingStrategy::new(&amm_pool, );
            let mut strategy = SnipeStrategy::new(ntoken1 as f64, ntoken0 as f64, slot);
            if let Some(swap_decision) = strategy.init_process().await {
                while let Err(e) = pool.execute_swap(swap_decision.amount_in, buy_direction).await {
                    info!("init swap error {e}");
                    failed_cnt += 1;
                    if failed_cnt > 4 {
                        let mut pools = POOL_MANAGER.pools.lock().await;
                        pools.remove(&amm_pool);
                        info!("pool {} end", amm_pool);
                        return
                    }
                }
            }
            info!("strategy init success!");
            loop {
                tokio::select! {
                    Some(step) = rx.recv() => {
                        pool.ntoken0 = step.token0 as u64;
                        pool.ntoken1 = step.token1 as u64;
                        let step = if sol_as_coin {
                            step
                        }
                        else {
                            Step {
                                from: step.from,
                                token0: step.token1,
                                token1: step.token0,
                                delta0: step.delta1,
                                delta1: step.delta0,
                                slot: step.slot
                            }
                        };
                        info!("{:?}" ,step);
                        match strategy.on_transaction(&step).await {
                            OnTransactionRet::SwapDecision(swap_decision) => {
                                let mut i = 0;
                                let swap_direction = if swap_decision.direction == DecisionDirection::BUY {
                                    buy_direction
                                }
                                else {
                                    sell_direction
                                };
                                while let Err(e) = pool.execute_swap(swap_decision.amount_in, swap_direction).await {
                                    error!("execute_swap {}th failed: {}", i, e);
                                    i += 1;
                                }
                            },
                            OnTransactionRet::End(is_end) => {
                                if is_end {
                                    break;
                                }
                            }
                        }
                    },
                    Ok(current_slot) = slot_rx.recv() => {
                        slot = current_slot;
                        match strategy.on_slot(slot).await {
                            OnTransactionRet::SwapDecision(swap_decision) => {
                                let mut i = 0;
                                let swap_direction = if swap_decision.direction == DecisionDirection::BUY {
                                    buy_direction
                                }
                                else {
                                    sell_direction
                                };
                                while let Err(e) = pool.execute_swap(swap_decision.amount_in, swap_direction).await {
                                    error!("execute_swap {}th failed: {}", i, e);
                                    i += 1;
                                }
                            },
                            OnTransactionRet::End(is_end) => {
                                if is_end {
                                    break;
                                }
                            }
                        }
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
    
    pub async fn execute_swap(&self, amount_in: u64, zero_for_one: bool) -> Result<(), Error> {
        if self.coin_mint == SOL && zero_for_one && amount_in > *MY_SOL.read().await as u64 {
            return Err(anyhow!("no enough wsol!"));
        }
        if self.pc_mint == SOL && !zero_for_one && amount_in > *MY_SOL.read().await as u64 {
            return Err(anyhow!("no enough wsol!"));
        }
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
        //info!("prepare to swap:{:?}", instructions);
        let sig = execute_tx_with_comfirm(&instructions).await?;
        info!("complete swap:{}", sig);
        // get tx detais here or wait for websocket?
        Ok(())
    }

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
            let target_orders = instruction.accounts[12];
            let market_program = instruction.accounts[15];
            let market = instruction.accounts[16];
            // maybe useful
            let creator = instruction.accounts[17];
            let creator_token0 = instruction.accounts[18];
            let creator_token1 = instruction.accounts[19];
            if coin_mint == SOL {
                let mut non_freeze_set = NON_FREEZE_TOKEN.write().await;
                if !non_freeze_set.remove(&pc_mint) {
                    info!("freeze coin:{pc_mint}");
                    return;
                }
            }
            else if pc_mint == SOL {
                let mut non_freeze_set = NON_FREEZE_TOKEN.write().await;
                if !non_freeze_set.remove(&coin_mint) {
                    info!("freeze coin:{pc_mint}");
                    return;
                }
            }
            else {
                return;
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
                    target_orders,
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
            //info!("init log{:?}", log);
        }
        LogType::Deposit => {
            let log: DepositLog = bincode::deserialize(&bytes).unwrap();
            let pool_address = instruction.accounts[1];
            let from = instruction.accounts[12];
            let step = Step {
                from,
                token0: log.pool_coin as f64,
                token1: log.pool_pc as f64,
                delta0: log.deduct_coin as f64,
                delta1: log.deduct_pc as f64,
                slot
            };
            POOL_MANAGER.on_step(&pool_address, step).await;
            //info!("deposit log{:?}", log);
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
                token0: log.pool_coin as f64,
                token1: log.pool_pc as f64,
                delta0: -(log.out_coin as f64),
                delta1: -(log.out_pc as f64),
                slot
            };
            POOL_MANAGER.on_step(&pool_address, step).await;
            //info!("withdraw log {:?}", log);
        }
        LogType::SwapBaseIn => {
            let log: SwapBaseInLog = bincode::deserialize(&bytes).unwrap();
            let pool_address = instruction.accounts[1];
            const ACCOUNT_LEN: usize = 17;
            let from = if instruction.accounts.len() == ACCOUNT_LEN + 1 {
                instruction.accounts[17]
            }
            else {
                instruction.accounts[16]
            };
            let (delta0, delta1) = if log.direction == Coin2PC as u64{
                (log.amount_in as f64, -(log.out_amount as f64))
            }
            else {
                (-(log.out_amount as f64), log.amount_in as f64)
            };
            let step = Step {
                from,
                token0: log.pool_coin as f64,
                token1: log.pool_pc as f64,
                delta0: delta0 as f64,
                delta1: delta1 as f64,
                slot
            };
            POOL_MANAGER.on_step(&pool_address, step).await;
            //info!("swap in log{:?}", log);
        }
        LogType::SwapBaseOut => {
            let log: SwapBaseOutLog = bincode::deserialize(&bytes).unwrap();
            let pool_address = instruction.accounts[1];
            const ACCOUNT_LEN: usize = 17;
            let from = if instruction.accounts.len() == ACCOUNT_LEN + 1 {
                instruction.accounts[17]
            }
            else {
                instruction.accounts[16]
            };
            let (delta0, delta1) = if log.direction == Coin2PC as u64{
                (log.deduct_in as f64, -(log.amount_out as f64))
            }
            else {
                (-(log.amount_out as f64), log.deduct_in as f64)
            };
            let step = Step {
                from,
                token0: log.pool_coin as f64,
                token1: log.pool_pc as f64,
                delta0,
                delta1,
                slot
            };
            POOL_MANAGER.on_step(&pool_address, step).await;
            //info!("swap out log{:?}", log);
        }
    }
}

pub async fn parse_raydium_transaction(instructions: &Vec<Instruction>, logs: &Vec<String>) -> Result<(), Error> {
    //info!("raydium tx:{:?}:{:?}", instructions, logs);
    if instructions.len() < logs.len() {
        error!("raydium tx:{:?}:{:?} size not comp", instructions, logs);
        return Err(anyhow!("size not comp"));
    }
    let mut j: usize = 0; // instruction iter
    for (i, log) in logs.iter().enumerate() {
        /*
        while j < instructions.len() {
            match AmmInstruction::unpack(&instructions[j].data) {
                Ok(instruction) => 
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
                Err(e) => {
                    error!("unpack error:{} {e}", j);
                    j += 1;
                    continue;
                }
            }
        }
        */
        if j >= instructions.len() {
            error!("raydium tx:{:?}:{:?} size not comp j = {}", instructions, logs, j);
            return Err(anyhow!("size not comp"));
        }
        decode_ray_log(&log[RAY_LOG_PREFIX.len()..], &instructions[j]).await;
        j += 1;
    }
    Ok(())
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
    else {
        error!("serum gen_vault_signer_key failed for data {:?}", data);
    }
}

lazy_static! {
    static ref NON_FREEZE_TOKEN: RwLock<HashSet<Pubkey>> = RwLock::new(HashSet::new());
}

pub async fn parse_spl_token_instruction(data: &[u8], accounts: &[u8], account_keys: &[Pubkey]) {
    match TokenInstruction::unpack(data) {
        Ok(instruction) => {
            match instruction {
                TokenInstruction::InitializeMint {
                    decimals,
                    mint_authority,
                    freeze_authority,
                } => {
                    if freeze_authority == COption::None {
                        let mut non_freeze_set = NON_FREEZE_TOKEN.write().await;
                        non_freeze_set.insert(account_keys[accounts[0] as usize]);
                    }
                }
                TokenInstruction::InitializeMint2 {
                    decimals,
                    mint_authority,
                    freeze_authority,
                } => {
                    if freeze_authority == COption::None {
                        let mut non_freeze_set = NON_FREEZE_TOKEN.write().await;
                        non_freeze_set.insert(account_keys[accounts[0] as usize]);
                    }
                }
                TokenInstruction::InitializeAccount => {
                }
                TokenInstruction::InitializeAccount2 { owner } => {                
                }
                TokenInstruction::InitializeAccount3 { owner } => {                    
                }
                TokenInstruction::InitializeMultisig { m } => {                    
                }
                TokenInstruction::InitializeMultisig2 { m } => {                    
                }
                TokenInstruction::Transfer { amount } => {                    
                }
                TokenInstruction::Approve { amount } => {                    
                }
                TokenInstruction::Revoke => {                    
                }
                TokenInstruction::SetAuthority {
                    authority_type,
                    new_authority,
                } => {
                    match authority_type {
                        AuthorityType::FreezeAccount => {
                            if new_authority == COption::None {
                                //info!("set freeze account none :{:?} {:?}", account_keys, account_keys);
                                let mut non_freeze_set = NON_FREEZE_TOKEN.write().await;
                                non_freeze_set.insert(account_keys[accounts[0] as usize]);
                            }
                        }
                        _ => ()
                    }
                }
                TokenInstruction::MintTo { amount } => {                    
                }
                TokenInstruction::Burn { amount } => {
                }
                TokenInstruction::CloseAccount => {                    
                }
                TokenInstruction::FreezeAccount => {                    
                }
                TokenInstruction::ThawAccount => {                    
                }
                TokenInstruction::TransferChecked { amount, decimals } => {     
                }
                TokenInstruction::ApproveChecked { amount, decimals } => {                    
                }
                TokenInstruction::MintToChecked { amount, decimals } => {                    
                }
                TokenInstruction::BurnChecked { amount, decimals } => {                    
                }
                TokenInstruction::SyncNative => {                    
                }
                TokenInstruction::GetAccountDataSize => {                    
                }
                TokenInstruction::InitializeImmutableOwner => {                    
                }
                TokenInstruction::AmountToUiAmount { amount } => {                    
                }
                TokenInstruction::UiAmountToAmount { ui_amount } => {                    
                }
                _ => {
                }
            }
        }
        Err(e) => error!("parse spl token instruction error:{e}")
    }
    
}