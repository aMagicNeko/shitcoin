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

        while let Some(transaction) = rx.recv().await {
            //info!("Processing transaction: {:?}", transaction);
            transaction.parse_transaction().await;
            /* 
            for address in addresses {
                if let Some(address_tx) = address_channels.get(&address) {
                    //if let Err(e) = address_tx.send(transaction.clone()).await {
                    //    error!("Failed to send transaction to address channel: {}", e);
                    //}
                } else {
                    let (address_tx, mut address_rx) = mpsc::channel(32);
                    address_channels.insert(address.clone(), address_tx);
                    let address_clone = address.clone();
                    task::spawn(async move {
                        while let Some(transaction) = address_rx.recv().await {
                            info!("Address {}: {:?}", address_clone, transaction);
                        }
                    });
                    /*
                    if let Some(address_tx) = address_channels.get(&address) {
                        if let Err(e) = address_tx.send(transaction.clone()).await {
                            error!("Failed to send transaction to new address channel: {}", e);
                        }
                    }
                    */
                }
            }
            */
        }
        info!("Router task ended");
    });
}