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
use tonic::transport::Channel;
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
use crate::raydium_amm::{MY_SOL, SOL};
use crate::priority_fee::{self, ACCOUNT_CREATE_PRIORITY_FEE_ESTIMATE, RAYDIUM_PRIORITY_FEE_ESTIMATE};
use anyhow::anyhow;
use solana_sdk::commitment_config::{CommitmentConfig, CommitmentLevel};
use spl_associated_token_account::{get_associated_token_address};
use spl_token::instruction::sync_native;
use spl_associated_token_account::instruction::create_associated_token_account;
use solana_client::rpc_request::TokenAccountsFilter;
use solana_account_decoder::{UiAccountEncoding, parse_token, UiAccountData, parse_account_data::{AccountAdditionalData, parse_account_data}};
use solana_transaction_status::parse_accounts::ParsedAccount;
use solana_sdk::commitment_config::{CommitmentLevel::Processed};
use solana_client::rpc_config::RpcSendTransactionConfig;
use crate::jito::{send_bundle_with_confirmation, send_bundle_no_wait, get_searcher_client_no_auth};
use jito_protos::searcher::searcher_service_client::SearcherServiceClient;
use tokio::sync::OnceCell;
use tokio::time::sleep;
use jito_protos::searcher::SubscribeBundleResultsRequest;
use spl_token::state::Mint;
pub static KEYPAIR: Lazy<Keypair> = Lazy::new(|| {
    read_keypair_file("./my-keypair.json").expect("Failed to read keypair file")
});

lazy_static! {
    static ref MIN_BALANCE_FOR_RENT_EXEMPTION: Arc<RwLock<u64>> = Arc::new(RwLock::new(0));
}

pub static RPC_CLIENT: Lazy<Arc<RpcClient>> = Lazy::new(|| {
    Arc::new(RpcClient::new("https://mainnet.helius-rpc.com/?api-key=e0f20dbd-b832-4a86-a74d-46c24db098b2"))
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
                        //info!("Updated block hash: {}", hash);
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
                                if mint == SOL {
                                    if let Some(sol_num) = account_info.get("tokenAmount").unwrap().get("amount").and_then(|v| v.as_str()) {
                                        let mut my_sol = MY_SOL.write().await;
                                        *my_sol = sol_num.parse::<u64>().unwrap() as f64;
                                        info!("init sol num:{sol_num}");
                                    }
                                }
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
    if None == TOKEN_VAULT_MAP.read().await.get(&SOL) || *MY_SOL.read().await < 1000000000.0 {
        let mut instructions = vec![];
        if None == TOKEN_VAULT_MAP.read().await.get(&SOL) {
            info!("start to create wsol account");
            let create_wsol_account_ix = create_associated_token_account(&user_pubkey, &user_pubkey, &SOL, &spl_token::id());
            instructions.push(create_wsol_account_ix);
        }
        let transfer_sol_ix = system_instruction::transfer(&user_pubkey, &wsol_account, 1_000_000_000); // 1 SOL
        let mut my_sol = MY_SOL.write().await;
        *my_sol += 1000000000.0;
        let sync_native_ix: Instruction = sync_native(&spl_token::id(), &wsol_account)?;
        instructions.push(transfer_sol_ix);
        instructions.push(sync_native_ix);
        info!("start to transfer sol to wsol");
        if let Some(block_hash) = *RECENT_BLOCKHASH.read().await {
            if let Some((priority_fee, _)) = *ACCOUNT_CREATE_PRIORITY_FEE_ESTIMATE.read().await {
                let compute_budget_instructions = vec![
                    //ComputeBudgetInstruction::set_compute_unit_price(priority_fee as u64),
                    ComputeBudgetInstruction::set_compute_unit_price(21222), // TODO: monitor this
                    ComputeBudgetInstruction::set_compute_unit_limit(80000),
                ];
                instructions.extend(compute_budget_instructions);
                let transaction = Transaction::new_signed_with_payer(
                    &instructions,
                    Some(&user_pubkey),
                    &[&KEYPAIR],
                    block_hash
                );
                info!("start to create wsol account 2");
                let result = RPC_CLIENT.send_and_confirm_transaction(&transaction);
                match result {
                    Ok(signature) => info!("create wsol account success: {:?}", signature),
                    Err(e) => { 
                        error!("create wsol account failed: {:?}", e);
                        return Err(anyhow!("create wsol account failed: {:?}", e))
                    }
                }
            }
        }
        else {
            error!("create wsol account failed: no block hash find");
            return Err(anyhow!("create wsol account failed: no block hash find"))
        }
    }
    TOKEN_VAULT_MAP.write().await.insert(SOL, wsol_account);
    info!("end init token account");
    Ok(())
}

pub async fn execute_tx_with_comfirm(instructions: &[Instruction]) -> Result<Signature, Error> {
    info!("start to execute_tx");
    if let Some((_, priority_fee)) = *RAYDIUM_PRIORITY_FEE_ESTIMATE.read().await {
        let mut my_priority_fee = priority_fee;
        if my_priority_fee >= 5000000.0 {
            my_priority_fee = 5000000.0;
        }
        let compute_budget_instructions = vec![
            ComputeBudgetInstruction::set_compute_unit_price(my_priority_fee as u64),
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
            //let commit = CommitmentConfig{commitment: Processed};
            let config = RpcSendTransactionConfig {
                skip_preflight: true,
                ..RpcSendTransactionConfig::default()
            };
        
            match RPC_CLIENT.send_transaction_with_config(&transaction, config) {
                Ok(sig) =>  {info!("processing transaction:{}", sig); Ok(sig)}
                Err(e) => Err(anyhow!("execute tx failed: {:?}", e))
            }
        }
        else {
            Err(anyhow!("create wsol account failed: no block hash find"))
        }
    }
    else {
        Err(anyhow!("no priorty_fee find"))
    }
}

static JITO_CLIENT: OnceCell<Arc<RwLock<SearcherServiceClient<Channel>>>> = OnceCell::const_new();

async fn initialize_jito_client() -> Arc<RwLock<SearcherServiceClient<Channel>>> {
    let client = get_searcher_client_no_auth("https://tokyo.mainnet.block-engine.jito.wtf")
        .await
        .expect("get jito client failed");
    Arc::new(RwLock::new(client))
}

async fn get_jito_client() -> Arc<RwLock<SearcherServiceClient<Channel>>> {
    JITO_CLIENT
        .get_or_init(initialize_jito_client)
        .await
        .clone()
}

pub fn start_bundle_results_loop() {
    task::spawn(async move {
        let client = get_jito_client().await;
        let _connection_errors: usize = 0;
        let mut response_errors: usize = 0;

        loop {
            sleep(Duration::from_millis(1000)).await;
            let stream = {
                let mut mut_client = client.write().await;
                mut_client.subscribe_bundle_results(SubscribeBundleResultsRequest {}).await
            };
            match stream {
                Ok(resp) => {
                    info!("boudle result: {:?}", resp.into_inner());
                }
                Err(e) => {
                    error!("searcher_bundle_results_error: {}", e);
                }
            }
        }
    });
}


pub async fn send_bundle() {
    let client = get_jito_client().await;
    
}
/*
pub fn send_bundle(rpc_url:, payer, message, num_txs, lamports, tip_account) {
    let client = get_searcher_client_no_auth(args.block_engine_url.as_str(),).await.expect("Failed to get searcher client with auth. Note: If you don't pass in the auth keypair, we can attempt to connect to the no auth endpoint");
    let payer_keypair = read_keypair_file(&payer).expect("reads keypair at path");

    // wait for jito-solana leader slot
    let mut is_leader_slot = false;
    while !is_leader_slot {
        let next_leader = client
            .get_next_scheduled_leader(NextScheduledLeaderRequest {
                regions: args.regions.clone(),
            })
            .await
            .expect("gets next scheduled leader")
            .into_inner();
        let num_slots = next_leader.next_leader_slot - next_leader.current_slot;
        is_leader_slot = num_slots <= 2;
        info!(
            "next jito leader slot in {num_slots} slots in {}",
            next_leader.next_leader_region
        );
        sleep(Duration::from_millis(500)).await;
    }

    // build + sign the transactions
    let blockhash = rpc_client
        .get_latest_blockhash()
        .await
        .expect("get blockhash");
    let txs: Vec<_> = (0..num_txs)
        .map(|i| {
            VersionedTransaction::from(Transaction::new_signed_with_payer(
                &[
                    build_memo(format!("jito bundle {i}: {message}").as_bytes(), &[]),
                    transfer(&payer_keypair.pubkey(), &tip_account, lamports),
                ],
                Some(&payer_keypair.pubkey()),
                &[&payer_keypair],
                blockhash,
            ))
        })
        .collect();

    send_bundle_with_confirmation(
        &txs,
        &rpc_client,
        &mut client,
        &mut bundle_results_subscription,
    )
    .await
    .expect("Sending bundle failed");
}
*/
pub fn query_spl_token_balance(mint: &Pubkey, least_amount: u64, tx: tokio::sync::mpsc::Sender<u64>) {
    let mint = mint.clone();
    task::spawn(async move {
        let vault_address = {
            let vault_map = TOKEN_VAULT_MAP.read().await;
            match vault_map.get(&mint) {
                Some(source_account) => *source_account,
                None => {
                    error!("query_spl_token_balance failed: not found mint vault{}", mint);
                    return ;
                }
            }
        };
        let mut retry_num = 0;
        while retry_num < 200 {
            retry_num += 1;
            let mint_decimals = match RPC_CLIENT.get_account_with_commitment(&mint, CommitmentConfig {commitment: CommitmentLevel::Processed}) {
                Ok(response) => {
                    if let Some(account) = response.value {
                        match Mint::unpack(&account.data) {
                            Ok(mint_data) => {
                                mint_data.decimals
                            }
                            Err(e) => {
                                error!("mint unpack error:{e}");
                                continue;
                            }
                        }
                    }
                    else {
                        continue;
                    }
                }
                Err(e) => {
                    error!("get_account_with_commitment error:{e}");
                    continue;
                }
            };
            match RPC_CLIENT.get_account_with_commitment(&vault_address, CommitmentConfig {commitment: CommitmentLevel::Processed}) {
                Ok(response) => {
                    if let Some(account_info) = response.value {
                        match parse_account_data(&vault_address, &spl_token::id(), &account_info.data, Some(AccountAdditionalData {spl_token_decimals: Some(mint_decimals)})) {
                            Ok(account) =>  {
                                if let Some(value) = account.parsed.get("info") {
                                    if let Some(val) = value.get("tokenAmount") {
                                        if let Some(amount_str) = val.get("amount").and_then(|v| v.as_str()) {
                                            if let Ok(amount) = amount_str.parse::<u64>() {
                                                info!("amount:{amount}");
                                                if amount > least_amount || retry_num == 199 {
                                                    tx.send(amount).await;
                                                    return;
                                                }
                                            }
                                        }

                                    }
                                }
                            }
                            Err(e) => {
                                error!("parsed account failed:{e}")
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("get_account failed:{e}")
                }
            }
        }
        error!("query_spl_token_balance failed for {mint}")
    });
}