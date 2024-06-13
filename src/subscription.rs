use tokio::sync::mpsc;
use tokio::task;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::protocol::Message;
use futures_util::{SinkExt, StreamExt};
use serde_json::{error, json};
use solana_transaction_status::{EncodedTransactionWithStatusMeta, UiTransactionStatusMeta, option_serializer::OptionSerializer, UiInstruction};
use log::{info, error};
use solana_sdk::{
    blake3::Hash, clock::Slot, commitment_config::{CommitmentConfig, CommitmentLevel}, instruction::CompiledInstruction, message::VersionedMessage::{Legacy, V0}, pubkey::{self, Pubkey}, signature::{self, Signature}
};
use tokio::sync::RwLock;
use std::{collections::{HashMap, VecDeque}};
use lazy_static::lazy_static;
use crate::raydium_amm::{RAY_AMM_ADDRESS, OPENBOOK_MARKET, parse_raydium_transaction, OpenbookInfo, OPENBOOK_MARKET_CACHE, parse_serum_instruction};
use std::str::FromStr;
use std::time::{Duration, Instant};
use tokio::time::{timeout};
use solana_client::{nonblocking::pubsub_client:: PubsubClient, rpc_response::{RpcBlockUpdate, SlotUpdate}};
use tokio::sync::broadcast;
use std::sync::Arc;
use solana_client::{
    rpc_config::{RpcBlockSubscribeConfig, RpcBlockSubscribeFilter},
    rpc_response,
};
use solana_transaction_status::{TransactionDetails, UiTransactionEncoding};
#[derive(Debug)]
pub struct Transaction {
    pub account_keys: Vec<Pubkey>,
    pub instructions: Vec<CompiledInstruction>,
    pub meta: UiTransactionStatusMeta
}
#[derive(Debug, PartialEq, Eq, PartialOrd)]
pub struct Instruction {
    pub out_index: usize,
    pub inner_index: i32,
    pub accounts: Vec<Pubkey>,
    pub data: Vec<u8>,
}

impl Ord for Instruction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let mut cmp = self.out_index.cmp(&other.out_index);
        if cmp == std::cmp::Ordering::Equal {
            cmp = self.inner_index.cmp(&other.inner_index);
        }
        cmp
    }
}

impl Transaction {
    pub async fn parse_transaction(&self) {
        let mut account_keys = self.account_keys.clone();
        if let OptionSerializer::Some(loaded_address) = &self.meta.loaded_addresses {
            for account_str in &loaded_address.writable {
                let account = Pubkey::from_str(account_str).unwrap();
                account_keys.push(account);
            }
            for account_str in &loaded_address.readonly {
                let account = Pubkey::from_str(account_str).unwrap();
                account_keys.push(account);
            }
        }
        let mut ray_amm_id: i32 = 256;
        let mut openbook_id: i32 = 257;
        let mut i = 0;
        for account in &account_keys {
            match account {
                &RAY_AMM_ADDRESS => ray_amm_id = i,
                &OPENBOOK_MARKET => openbook_id = i,
                _ => (),
            };
            i += 1
        }
        let mut ray_logs: Vec<String> = Vec::new();
        if let OptionSerializer::Some(logs) = &self.meta.log_messages {
            for log in logs {
                if log.as_str().starts_with("Program log: ray_log:") {
                    ray_logs.push(log.to_string()); // no use logs vec anymore
                }
            }
        }
        let mut ray_instructions: Vec<Instruction> = Vec::new();
        for (i, instruction) in self.instructions.iter().enumerate() {
            match instruction.program_id_index {
                id if id == ray_amm_id as u8  => {
                    let mut accounts: Vec<Pubkey> = Vec::with_capacity(instruction.accounts.len());
                    for index in &instruction.accounts {
                        accounts.push(account_keys[*index as usize]);
                    }
                    ray_instructions.push(Instruction{out_index: i, inner_index: -1, accounts, data: instruction.data.clone()});
                }
                id if id == openbook_id as u8 => {
                    parse_serum_instruction(&instruction.data, &instruction.accounts, &account_keys).await;
                }
                _ => ()
            }
        }
        if let OptionSerializer::Some(inner_instructions) = &self.meta.inner_instructions {
            for instructions in inner_instructions.iter() {
                for (o, uniinstruction) in instructions.instructions.iter().enumerate() {
                    if let UiInstruction::Compiled(instruction) = uniinstruction {
                        if instruction.program_id_index == ray_amm_id as u8 {
                            let mut accounts: Vec<Pubkey> = Vec::with_capacity(instruction.accounts.len());
                            for index in &instruction.accounts {
                                accounts.push(account_keys[*index as usize]);
                            }
                            ray_instructions.push(Instruction{out_index: instructions.index as usize, inner_index: o as i32, accounts: accounts, data: instruction.data.as_bytes().to_vec()});
                        }
                    }
                    else {
                        error!("Not implement kind of inner instruction!");
                    }
                }
            }
        }
        if ray_logs.len() != 0 && ray_amm_id != 256 {
            ray_instructions.sort_unstable();
            if let Err(e) = parse_raydium_transaction(&ray_instructions, &ray_logs).await {
                error!("error tx:{:?}", self);
            }
        }
    }
}

struct TransactionFilter {
    old_slot: HashMap<String, ()>,
    prev_slot: HashMap<String, ()>,
    cur_slot: HashMap<String, ()>,
    prev_update_time: Instant,
}

impl TransactionFilter {
    pub fn new() -> Self {
        Self {
            old_slot: HashMap::new(),
            prev_slot: HashMap::new(),
            cur_slot: HashMap::new(),
            prev_update_time: Instant::now(),
        }
    }

    pub fn record_transaction(&mut self, signature: &str) -> bool {
        let now = Instant::now();
        if now.duration_since(self.prev_update_time) > Duration::from_secs(3) {
            // Rotate the slots
            self.old_slot = std::mem::take(&mut self.prev_slot);
            self.prev_slot = std::mem::take(&mut self.cur_slot);
            self.prev_update_time = now;
        }

        if self.cur_slot.contains_key(signature) || self.prev_slot.contains_key(signature) || self.old_slot.contains_key(signature) {
            return false;
        }

        self.cur_slot.insert(signature.to_string(), ());
        true
    }
}

pub fn start_subscription(tx: mpsc::Sender<Transaction>) {
    task::spawn(async move {
        info!("Starting subscription task");
        let url = "wss://atlas-mainnet.helius-rpc.com/?api-key=e0f20dbd-b832-4a86-a74d-46c24db098b2";
        loop { // might lose some data
        let timeout_duration = Duration::from_secs(5);
        match timeout(timeout_duration, connect_async(url)).await {
            Ok(Ok((mut ws_stream, _))) => {
                let params = json!([
                    {
                        "vote": false,
                        "failed": false,
                        "accountInclude": ["675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8", "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4", "srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX"]
                        //"accountInclude": ["HWy1jotHpo6UqeQxx49dpYYdQB8wj9Qk9MdxwjLvDHB8", "82iPEvGiTceyxYpeLK3DhSwga3R5m4Yfyoydd13CukQ9"]
                    },
                    {
                        "commitment": "processed",
                        "encoding": "base64",
                        "transaction_details": "full",
                        "showRewards": false,
                        "maxSupportedTransactionVersion": 0
                    }
                ]);

                let request = json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "transactionSubscribe",
                    "params": params
                });

                if let Err(e) = ws_stream.send(Message::Text(request.to_string())).await {
                    error!("Failed to send request: {}", e);
                    return;
                }
                info!("Subscription request sent successfully");
                let mut transaction_filter = TransactionFilter::new();
                while let Ok(Some(message)) = timeout(timeout_duration,  ws_stream.next()).await {
                    //info!("{:?}", message);
                    match message {
                        Ok(Message::Text(text)) => {
                            if let Ok(response) = serde_json::from_str::<serde_json::Value>(&text) {

                                if let Some(result) = response["params"]["result"].as_object() {
                                    if let Some(signature) = result.get("signature") {
                                        if !transaction_filter.record_transaction(&signature.to_string()) {
                                            continue
                                        }
                                    }
                                    if let Some(transaction_value) = result.get("transaction") {
                                        match serde_json::from_value::<EncodedTransactionWithStatusMeta>(transaction_value.clone()) { 
                                            Ok(encoded_transaction) => {
                                                if let Some(meta) = &encoded_transaction.meta {
                                                    if let Some(transaction) = &encoded_transaction.transaction.decode() {
                                                        let (account_keys, instructions) = match &transaction.message {
                                                            Legacy(message) => (&message.account_keys, &message.instructions),
                                                            V0(message) => (&message.account_keys, &message.instructions),
                                                        };
                                                        let tx_with_meta = Transaction {
                                                            account_keys: account_keys.to_vec(),
                                                            instructions: instructions.to_vec(),
                                                            meta: meta.clone(),
                                                        };
                                                        if let Err(e) = tx.send(tx_with_meta).await {
                                                            error!("Failed to send transaction: {}", e);
                                                        }
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                error!("Failed to parse transaction: {:?}", e);
                                                error!("Transaction value: {:?}", transaction_value);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("Error reading message: {}", e);
                        }
                        _ => (),
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("Error while connecting: {}", e);
            },
            Err(e) => {
                error!("Failed to connect to Solana WebSocket: {}", e);
            }
        }
    }
    });
}

lazy_static! {
    pub static ref SLOT_BROADCAST_CHANNEL: (broadcast::Sender<u64>, broadcast::Receiver<u64>) = broadcast::channel(16);
}
lazy_static! {
    pub static ref CURRENT_SLOT: Arc<RwLock<u64>> = Arc::new(RwLock::new(0));
}
// slot update subscription loop that attempts to maintain a connection to an RPC server
pub fn start_slot_subscription() {
    let pubsub_addr = "wss://mainnet.helius-rpc.com/?api-key=e0f20dbd-b832-4a86-a74d-46c24db098b2";
    task::spawn(async move {
    info!("Starting subscription task");
    loop {
        let tx = SLOT_BROADCAST_CHANNEL.0.clone();
        match PubsubClient::new(&pubsub_addr).await {
            Ok(pubsub_client) => match pubsub_client.slot_updates_subscribe().await {
                Ok((mut slot_update_subscription, _unsubscribe_fn)) => {
                    let timeout_duration = Duration::from_secs(1);
                    while let Ok(Some(slot_update)) = timeout(timeout_duration, slot_update_subscription.next()).await {
                        if let SlotUpdate::FirstShredReceived { slot, timestamp: _ } = slot_update {
                            //info!("slot_subscribe_slot{}", slot);
                            let mut current_slot = CURRENT_SLOT.write().await;
                            *current_slot = slot;
                            tx.send(slot).unwrap();
                        }
                    }
                }
                Err(e) => {
                    error!("slot_subscribe_error {}", e.to_string());
                }
            },
            Err(e) => {
                error!("slot_subscribe_pubsub_connect_error {}",e.to_string());
            }
        }
    }
});
}

pub fn block_subscribe_loop() {
    task::spawn(async move {
    loop {
        match PubsubClient::new("wss://mainnet.helius-rpc.com/?api-key=e0f20dbd-b832-4a86-a74d-46c24db098b2").await {
            Ok(pubsub_client) => match pubsub_client
                .block_subscribe(
                    RpcBlockSubscribeFilter::All,
                    Some(RpcBlockSubscribeConfig {
                        commitment: Some(CommitmentConfig {
                            commitment: CommitmentLevel::Processed,
                        }),
                        encoding: Some(UiTransactionEncoding::Base64),
                        transaction_details: Some(TransactionDetails::None),
                        show_rewards: Some(false),
                        max_supported_transaction_version: None,
                    }),
                )
                .await
            {
                Ok((mut block_update_subscription, _unsubscribe_fn)) => {
                    while let Some(block_update) = block_update_subscription.next().await {
                        if let Some(block) = block_update.value.block {
                            info!("block hash {}", block.blockhash);
                        }
                    }
                }
                Err(e) => {
                    error!("block subscribe error {e}");
                }
            },
            Err(e) => {
                error!("block subscribe error {e}");
            }
        }
    }
});
}