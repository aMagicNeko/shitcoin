use crate::{subscription::Transaction, 
    raydium_amm::RAY_AMM_ADDRESS};
use tokio::sync::mpsc;
use tokio::task;
use std::collections::HashMap;
use log::{info, error};

pub fn start_router(mut rx: mpsc::Receiver<Transaction>) {
    task::spawn(async move {
        info!("Starting router task");
        let mut address_channels: HashMap<String, mpsc::Sender<Transaction>> = HashMap::new();

        loop {
            tokio::select! {
                Some(transaction) = rx.recv() => {
                    transaction.parse_transaction().await;
                },
                else => error!("channels closed"),
            }
        }
    });
}