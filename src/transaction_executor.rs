use anyhow::Error;
use serum_dex::instruction;
use solana_program::pubkey;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signer::Signer;
use solana_sdk::system_instruction;
use solana_sdk::signature::{read_keypair_file, Keypair, Signature};
use solana_sdk::instruction::Instruction;
use solana_sdk::transaction::Transaction;
use solana_sdk::hash::Hash;
use solana_sdk::compute_budget::ComputeBudgetInstruction;
use once_cell::sync::Lazy;
use std::str::FromStr;
use std::sync::Arc;
use spl_token::instruction::{initialize_account, mint_to, transfer};
use spl_token::state::Account;
use lazy_static::lazy_static;
use tokio::{sync::{Mutex, RwLock}, task, time::{interval, Duration}};
use solana_client::rpc_client::RpcClient;
use solana_sdk::program_pack::Pack;
use std::collections::HashMap;
use log::{info, error};
use crate::subscription::SLOT_BROADCAST_CHANNEL;
use crate::raydium_amm::SOL;
use anyhow::anyhow;
use spl_associated_token_account::{get_associated_token_address};
use spl_token::instruction::sync_native;
use spl_associated_token_account::instruction::create_associated_token_account;
use solana_client::rpc_request::TokenAccountsFilter;
use solana_account_decoder::{UiAccountEncoding, parse_token, UiAccountData};
use solana_transaction_status::parse_accounts::ParsedAccount;

pub static KEYPAIR: Lazy<Keypair> = Lazy::new(|| {
    read_keypair_file("./my-keypair.json").expect("Failed to read keypair file")
});

lazy_static! {
    static ref MIN_BALANCE_FOR_RENT_EXEMPTION: Arc<RwLock<u64>> = Arc::new(RwLock::new(0));
}

pub static RPC_CLIENT: Lazy<Arc<RpcClient>> = Lazy::new(|| {
    Arc::new(RpcClient::new("https://devnet.helius-rpc.com/?api-key=e0f20dbd-b832-4a86-a74d-46c24db098b2"))
});

lazy_static! {
    pub static ref TOKEN_VAULT_MAP: Arc<RwLock<HashMap<Pubkey, Pubkey>>> = Arc::new(RwLock::new(HashMap::new()));
}

pub fn start_fetch_minimum_balance_loop() {
    task::spawn(async move {
        let mut interval = interval(Duration::from_secs(600));
        loop {
            match RPC_CLIENT.get_minimum_balance_for_rent_exemption(Account::LEN) {
                Ok(min_balance) => {
                    let mut balance = MIN_BALANCE_FOR_RENT_EXEMPTION.write().await;
                    *balance = min_balance;
                    info!("Updated minimum balance for rent exemption: {}", min_balance);
                },
                Err(e) => error!("Failed to fetch minimum balance for rent exemption: {:?}", e),
            }
            interval.tick().await;
        }
    });
}

lazy_static::lazy_static! {
    static ref RECENT_BLOCKHASH: Arc<RwLock<Option<Hash>>> = Arc::new(RwLock::new(None));
}

pub fn start_get_block_hash_loop() {
    task::spawn(async move {
        let mut slot_rx = SLOT_BROADCAST_CHANNEL.0.subscribe();
        loop {
            if let Ok(slot) = slot_rx.recv().await {
                match RPC_CLIENT.get_latest_blockhash() {
                    Ok(hash) => {
                        let mut old_hash = RECENT_BLOCKHASH.write().await;
                        *old_hash = Some(hash);
                        info!("Updated block hash: {}", hash);
                    },
                    Err(e) => error!("Failed to fetch recent block hash: {:?}", e),
                }
            }
            else {
                error!("block hash loop error: slot rx recv error!");
            }
        }
    });
}

pub async fn gen_associated_token_account(token_mint: &Pubkey, payer: &Pubkey) -> Result<(Vec<Instruction>, Pubkey), Error> {
    let user_token_destination = get_associated_token_address(payer, token_mint);
    let mut instructions = vec![];
    // 创建新的代币账户
    let create_associated_account_ix = create_associated_token_account(
        payer,
        payer,
        token_mint,
        &spl_token::id()
    );
    instructions.push(create_associated_account_ix);
    TOKEN_VAULT_MAP.write().await.insert(token_mint.clone(), user_token_destination);
    Ok((instructions, user_token_destination))
}

pub async fn send_transaction(instructions: &[Instruction], ) -> Result<(), Error> {
    let pubkey = KEYPAIR.pubkey();
    if let Some(block_hash) =  RECENT_BLOCKHASH.read().await.clone() {
        let transaction = Transaction::new_signed_with_payer(
            instructions,
            Some(&pubkey),
            &[&*KEYPAIR],
            block_hash,
        );
        match RPC_CLIENT.send_and_confirm_transaction(&transaction) {
            Ok(signature) => info!("Transaction successful with signature: {:?}", signature),
            Err(e) => {
                error!("Transaction failed: {:?}", e);
                return Err(anyhow!("transaction failed: {:?}", e));
            }
        }
    }
    Ok(())
}

pub async fn init_token_account() -> Result<(), Error>{
    let response = RPC_CLIENT.get_token_accounts_by_owner(
        &KEYPAIR.pubkey(),
        TokenAccountsFilter::ProgramId(spl_token::id()),
    )?;
    for token_account in response.iter() {
        info!("{:?}", token_account);
        let account_data = &token_account.account.data;
        match account_data {
            UiAccountData::Json(parsed_data) => {
                if let serde_json::Value::Object(parsed_account) = &parsed_data.parsed {
                    //info!("Parsed account: {:?}", parsed_account);
                    if let Some(account_info) = parsed_account.get("info") {
                        if let Some(is_native) = account_info.get("isNative").and_then(|v| v.as_bool()) {
                            if is_native {
                                let mint_str = account_info.get("mint").and_then(|v| v.as_str()).unwrap_or("");
                                let mint = match Pubkey::from_str(mint_str) {
                                    Ok(mint) => mint,
                                    Err(e) => {
                                        error!("Failed to parse mint pubkey: {}", e);
                                        continue;
                                    }
                                };
                                let associate_account = match Pubkey::from_str(&token_account.pubkey) {
                                    Ok(pubkey) => pubkey,
                                    Err(e) => {
                                        error!("Failed to parse associate account pubkey: {}", e);
                                        continue;
                                    }
                                };
                                TOKEN_VAULT_MAP.write().await.insert(mint, associate_account);
                                info!("Inserted into TOKEN_VAULT_MAP: mint = {}, associate_account = {}", mint, associate_account);
                            }
                        }
                    }
                } else {
                    error!("Failed to parse account from JSON.");
                }
            },
            _ => {
                error!("Unsupported account data format.");
            }
        }
    }
    let user_pubkey = KEYPAIR.pubkey();
    let wsol_account = get_associated_token_address(&user_pubkey, &SOL);
    if let None = TOKEN_VAULT_MAP.read().await.get(&SOL) {
        let create_wsol_account_ix = create_associated_token_account(&user_pubkey, &user_pubkey, &SOL, &spl_token::id());
        let transfer_sol_ix = system_instruction::transfer(&user_pubkey, &wsol_account, 1_000_000_000); // 1 SOL
        let sync_native_ix = sync_native(&spl_token::id(), &wsol_account)?;
        let instructions = vec![
            create_wsol_account_ix,
            transfer_sol_ix,
            sync_native_ix,
        ];
        if let Some(block_hash) = *RECENT_BLOCKHASH.read().await {
            let transaction = Transaction::new_signed_with_payer(
                &instructions,
                Some(&user_pubkey),
                &[&KEYPAIR],
                block_hash
            );
            let result = RPC_CLIENT.send_and_confirm_transaction(&transaction);
            match result {
                Ok(signature) => info!("create wsol account success: {:?}", signature),
                Err(e) => { 
                    error!("create wsol account failed: {:?}", e);
                    return Err(anyhow!("create wsol account failed: {:?}", e))
                }
            }
        }
        else {
            error!("create wsol account failed: no block hash find");
            return Err(anyhow!("create wsol account failed: no block hash find"))
        }
    }
    TOKEN_VAULT_MAP.write().await.insert(SOL, wsol_account);
    Ok(())
}

pub async fn execute_tx_with_comfirm(instructions: &[Instruction]) -> Result<Signature, Error> {
    let compute_budget_instructions = vec![
        ComputeBudgetInstruction::set_compute_unit_price(2122),
        //ComputeBudgetInstruction::set_compute_unit_price(21222), // TODO: monitor this
        ComputeBudgetInstruction::set_compute_unit_limit(80000),
    ];
    let mut all_instructions = Vec::new();
    all_instructions.extend_from_slice(&compute_budget_instructions);
    all_instructions.extend_from_slice(instructions);
    if let Some(block_hash) = *RECENT_BLOCKHASH.read().await {
        let transaction = Transaction::new_signed_with_payer(
            &all_instructions,
            Some(&KEYPAIR.pubkey()),
            &[&KEYPAIR],
            block_hash
        );
        match RPC_CLIENT.send_and_confirm_transaction(&transaction) {
            Ok(sig) => Ok(sig),
            Err(e) => Err(anyhow!("execute tx failed: {:?}", e))
        }
    }
    else {
        Err(anyhow!("create wsol account failed: no block hash find"))
    }
}