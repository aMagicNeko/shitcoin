use solana_sdk::{pubkey::Pubkey, signer::Signer};
use solana_sdk::pubkey;
use log::{info, error};
use crate::transaction_executor::KEYPAIR;
use tokio::sync::Mutex;
use std::collections::HashMap;
use tokio::sync::RwLock;
use tokio::sync::mpsc::{self, Sender, Receiver};
use tokio::task;
use lazy_static::lazy_static;
use crate::subscription::{SLOT_BROADCAST_CHANNEL, CURRENT_SLOT, unregister_sig_subscription, register_sig_subscription};
use crate::subscription::TransactionResult;
use solana_sdk::signature::Signature;
use crate::transaction_executor::query_spl_token_balance;
pub const PUMP_FUN_PROGRAM: Pubkey = pubkey!("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P");
pub const PUMP_FEE_RECIPIENT: Pubkey = pubkey!("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM");
const BUY_UINT: u64 = 16927863322537952870;
const SELL_UINT: u64 = 12502976635542562355;
const INIT_TOKEN: u64 = 1_000_000_000_000_000_000;
const INIT_SOL: u64 = 16_000_000;
#[derive(Debug)]
struct PumpPool {
    pub global: Pubkey,
    pub fee_recipient: Pubkey,
    pub mint: Pubkey,
    pub bonding_curve: Pubkey,
    pub associated_bonding_curve: Pubkey,
    pub associated_user: Pubkey,
    pub user: Pubkey,
    pub system_program: Pubkey,
    pub token_program: Pubkey,
    pub rent: Pubkey,
    pub event_authority: Pubkey,
    pub program: Pubkey
}

impl PumpPool {
        pub fn new(global: Pubkey,
            fee_recipient: Pubkey,
            mint: Pubkey,
            bonding_curve: Pubkey,
            associated_bonding_curve: Pubkey,
            associated_user: Pubkey,
            user: Pubkey,
            system_program: Pubkey,
            token_program: Pubkey,
            rent: Pubkey,
            event_authority: Pubkey,
            program: Pubkey,
            init_buy: u64
        ) -> Sender<Step> {
            let (tx, mut rx): (Sender<Step>, Receiver<Step>) = mpsc::channel(100);
            task::spawn(async move {
                let mut pool = PumpPool {
                    global,
                    fee_recipient,
                    mint,
                    bonding_curve,
                    associated_bonding_curve,
                    associated_user,
                    user,
                    system_program,
                    token_program,
                    rent,
                    event_authority,
                    program
                };
                info!("new pump {:?}", pool);
                let mut slot_rx = SLOT_BROADCAST_CHANNEL.0.subscribe();
                let mut slot: u64 = *CURRENT_SLOT.read().await;
                let (sig_tx, mut sig_rx) = mpsc::channel::<TransactionResult>(100);
                let (token_query_tx, mut token_query_rx) = mpsc::channel::<u64>(10);
                let mut pending_sell: Option<Signature> = None;
                let mut pool_token = INIT_TOKEN - init_buy;
                let liquidity = INIT_TOKEN as u128 * INIT_SOL as u128;
                loop {
                    tokio::select! {
                        Some(step) = rx.recv() => {
                            info!("{:?}" ,step);
                            match step.direction {
                                BUY => {
                                    pool_token += step.token;
                                }
                                SELL => {
                                    pool_token -= step.token;
                                }
                            }
                            if pending_sell == None {
                                let pool_sol = liquidity / pool_token as u128;
                                let sol_out = pool_sol - liquidity / (pool_token + init_buy) as u128;
                                if sol_out > 50_000_000 {

                                }
                            }
                        },
                        Ok(current_slot) = slot_rx.recv() => {
                            slot = current_slot;
                        }
                        Some(tx_res) = sig_rx.recv() => {
                            match tx_res {
                                TransactionResult::SUCCESS => {
                                    match pending_sell {
                                        Some(sig) => {
                                            unregister_sig_subscription(&sig).await;
                                            pending_sell = None;
                                            break;
                                        },
                                        None => {
                                            error!("no pending tx exist while receiving sig result");
                                        }
                                    }
                                },
                                TransactionResult::FAIL => {
                                    match pending_sell {
                                        Some(sig) => {
                                            unregister_sig_subscription(&sig).await;
                                            pending_sell = None;
                                            error!("executed tx failed!");
                                            query_spl_token_balance(&mint, 0, token_query_tx.clone());
                                        },
                                        None => {
                                            error!("no pending tx exist while receiving sig result");
                                        }
                                    }
                                }
                            }
                        }
                        Some(position) = token_query_rx.recv() => {
                            
                        }
                        else => error!("channel closed"),
                    }
                }
                let mut pools = PUMPPOOL_MANAGER.write().await;
                pools.remove(&mint);
                info!("pool {} end", pool);
                return
            });
            tx
        }
        
        pub async fn execute_swap(&self, amount_in: u64, zero_for_one: bool) -> Result<Signature, Error> {
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
            Ok(sig)
        }
    
    }

#[derive(Debug)]
pub enum Direction {
    BUY = 1,
    SELL = 2
}
#[derive(Debug)]
pub struct Step {
    pub from: Pubkey,
    pub direction: Direction,
    pub token: u64,
}

lazy_static! {
    static ref PUMPPOOL_MANAGER: RwLock<HashMap<Pubkey, Sender<Step>>> = RwLock::new(HashMap::new());
}


fn parse_instruction_data(data: &[u8]) -> (u64, u64, u64) {
    use std::convert::TryInto;

    let part1 = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let part2 = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let part3 = u64::from_le_bytes(data[16..24].try_into().unwrap());

    (part1, part2, part3)
}


pub async fn parse_spl_token_instruction(data: &[u8], accounts: &[u8], account_keys: &[Pubkey]) {
    let (p1, p2, p3) = parse_instruction_data(data);
    match p1 {
        BUY_UINT => {
            let global = account_keys[accounts[0] as usize];
            let fee_recipient = account_keys[accounts[1] as usize];
            let mint = account_keys[accounts[2] as usize];
            let bonding_curve = account_keys[accounts[3] as usize];
            let associated_bonding_curve = account_keys[accounts[4] as usize];
            let associated_user = account_keys[accounts[5] as usize];
            let user = account_keys[accounts[6] as usize];
            let system_program = account_keys[accounts[7] as usize];
            let token_program = account_keys[accounts[8] as usize];
            let rent = account_keys[accounts[9] as usize];
            let event_authority = account_keys[accounts[10] as usize];
            let program = account_keys[accounts[11] as usize];
            if user == KEYPAIR.pubkey() {
                let init_buy = p2;
                let tx = PumpPool::new(global,
                    fee_recipient,
                    mint,
                    bonding_curve,
                    associated_bonding_curve,
                    associated_user,
                    user,
                    system_program,
                    token_program,
                    rent,
                    event_authority,
                    program,
                    init_buy);
                PUMPPOOL_MANAGER.write().await.insert(mint, tx);
            }
            else {
                if let Some(tx) = PUMPPOOL_MANAGER.read().await.get(&mint) {
                    let step = Step {
                        from: user,
                        direction: Direction::BUY,
                        token: p2
                    };
                    tx.send(step).await;
                }
            }
        }
        SELL_UINT => {
            let global = account_keys[accounts[0] as usize];
            let fee_recipient = account_keys[accounts[1] as usize];
            let mint = account_keys[accounts[2] as usize];
            let bonding_curve = account_keys[accounts[3] as usize];
            let associated_bonding_curve = account_keys[accounts[4] as usize];
            let associated_user = account_keys[accounts[5] as usize];
            let user = account_keys[accounts[6] as usize];
            let system_program = account_keys[accounts[7] as usize];
            let token_program = account_keys[accounts[8] as usize];
            let rent = account_keys[accounts[9] as usize];
            let event_authority = account_keys[accounts[10] as usize];
            let program = account_keys[accounts[11] as usize];
            let pool_manager = PUMPPOOL_MANAGER.read().await;
            if let Some(tx) = pool_manager.get(&mint) {
                let step = Step {
                    from: user,
                    direction: Direction::SELL,
                    token: p2
                };
                tx.send(step).await;
            }
        }
        _ => {
            info!("no match p1{p1}");
        }
    }
}

