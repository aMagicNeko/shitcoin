use solana_client::{rpc_client::RpcClient, rpc_request::RpcRequest};
use solana_sdk::pubkey::Pubkey;
use tokio::task;
use tokio::sync::RwLock;
use std::sync::Arc;
use once_cell::sync::Lazy;
use anyhow::Error;
use log::{info, error};
use serde_json::{json, Value};
use crate::raydium_amm::RAY_AMM_ADDRESS;
use crate::subscription::SLOT_BROADCAST_CHANNEL;

pub static RPC_CLIENT: Lazy<Arc<RpcClient>> = Lazy::new(|| {
    Arc::new(RpcClient::new("https://mainnet.helius-rpc.com/?api-key=e0f20dbd-b832-4a86-a74d-46c24db098b2".to_string()))
});

pub static ACCOUNT_CREATE_PRIORITY_FEE_ESTIMATE: Lazy<Arc<RwLock<Option<(f64, f64)>>>> = Lazy::new(|| {
    Arc::new(RwLock::new(None))
});

pub static RAYDIUM_PRIORITY_FEE_ESTIMATE: Lazy<Arc<RwLock<Option<(f64, f64)>>>> = Lazy::new(|| {
    Arc::new(RwLock::new(None))
});

pub fn start_get_priority_fee_estimate_loop() {
    task::spawn(async move {
        let mut slot_rx = SLOT_BROADCAST_CHANNEL.0.subscribe();
        loop {
            if let Ok(_slot) = slot_rx.recv().await {
                match get_priority_fee_estimate(&spl_token::id()).await {
                    Ok((mid, high)) => {
                        let mut fee_estimate = ACCOUNT_CREATE_PRIORITY_FEE_ESTIMATE.write().await;
                        *fee_estimate = Some((mid, high));
                        //info!("Updated priority account fee estimate: mid = {}, high = {}", mid, high);
                    },
                    Err(e) => error!("Failed to fetch priority fee estimate: {:?}", e),
                }

                match get_priority_fee_estimate(&RAY_AMM_ADDRESS).await {
                    Ok((mid, high)) => {
                        let mut fee_estimate = RAYDIUM_PRIORITY_FEE_ESTIMATE.write().await;
                        *fee_estimate = Some((mid, high));
                        //info!("Updated priority raydium fee estimate: mid = {}, high = {}", mid, high);
                    },
                    Err(e) => error!("Failed to fetch priority fee estimate: {:?}", e),
                }
            }
        }
    });
}

async fn get_priority_fee_estimate(pubkey: &Pubkey) -> Result<(f64, f64), Error> {
    let client = RPC_CLIENT.clone();

    let params = json!([[pubkey.to_string()],
        ]);

    let request = RpcRequest::GetRecentPrioritizationFees;

    let response: Value = client.send(request, params).map_err(|e| {
        error!("RPC error: {:?}", e);
        e
    })?;
    //info!("{}", response);

    let mut fees: Vec<f64> = response.as_array()
        .ok_or_else(|| anyhow::anyhow!("Failed to parse priority fee estimate"))?
        .iter()
        .filter_map(|entry| entry.get("prioritizationFee").and_then(|fee| fee.as_f64()))
        .collect();

    fees.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    while fees.len() < 5 {
        fees.push(0.0);
    }

    let recent_fees: Vec<f64> = fees.iter().rev().take(5).cloned().collect();
    let mid = calculate_average(&recent_fees).ok_or_else(|| anyhow::anyhow!("Failed to calculate average"))?;
    let high = *recent_fees.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).ok_or_else(|| anyhow::anyhow!("Failed to find max value"))?;

    Ok((mid, high))
}

fn calculate_average(fees: &[f64]) -> Option<f64> {
    if fees.is_empty() {
        return None;
    }
    let sum: f64 = fees.iter().sum();
    Some(sum / fees.len() as f64)
}
