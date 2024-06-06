use anyhow::{Result, anyhow};
use futures_util::StreamExt;
use solana_client::{nonblocking::pubsub_client:: PubsubClient, rpc_response::{RpcBlockUpdate, SlotUpdate}};
use solana_sdk::{pubkey::Pubkey,
    commitment_config::{CommitmentConfig, CommitmentLevel}
};
use tokio::time::{timeout, Duration, sleep};
use solana_client::rpc_config::{RpcBlockSubscribeConfig, RpcBlockSubscribeFilter};
use solana_transaction_status::{UiTransactionEncoding, TransactionDetails};
use solana_metrics::{datapoint_error, datapoint_info};

pub struct DataFetcher {
    pubsub_client: PubsubClient,
}

impl DataFetcher {
    pub async fn new(url: &str) -> Result<Self> {
        let pubsub_client = match timeout(Duration::from_secs(10), PubsubClient::new(url)).await {
            Ok(client) => match client {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to create PubsubClient: {:?}", e);
                    return Err(e.into());
                }
            },
            Err(_) => {
                eprintln!("Connection timed out");
                return Err(anyhow!("Connection timed out"));
            }
        };

        Ok(Self { pubsub_client })
    }

    pub async fn subscribe_blocks(&self, filter_str: &str) -> Result<()> {
        let filter = RpcBlockSubscribeFilter::MentionsAccountOrProgram("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8".to_string());
        /*
        let filter = match filter_str {
            "" => RpcBlockSubscribeFilter::All,
            _ => RpcBlockSubscribeFilter::MentionsAccountOrProgram(filter_str.to_string())
        };
        */
        let config = RpcBlockSubscribeConfig {
            //commitment: Some(CommitmentConfig::confirmed()),
            commitment: Some(CommitmentConfig::finalized()),
            encoding: Some(UiTransactionEncoding::JsonParsed),
            transaction_details: Some(TransactionDetails::Full),
            show_rewards: Some(false),
            max_supported_transaction_version: None,
        };
        let (mut block_stream, _unsubscribe) = self.pubsub_client.block_subscribe(filter, Some(config)).await?;
        while let Some(block_info) = block_stream.next().await {
            print!("{:?}", block_info.value);
        }
        Ok(())
    }
}


// slot update subscription loop that attempts to maintain a connection to an RPC server
pub async fn slot_subscribe_loop(pubsub_addr: String) {
    let mut connect_errors: u64 = 0;
    let mut slot_subscribe_errors: u64 = 0;
    let mut slot_subscribe_disconnect_errors: u64 = 0;

    loop {
        sleep(Duration::from_secs(1)).await;

        match PubsubClient::new(&pubsub_addr).await {
            Ok(pubsub_client) => match pubsub_client.slot_updates_subscribe().await {
                Ok((mut slot_update_subscription, _unsubscribe_fn)) => {
                    while let Some(slot_update) = slot_update_subscription.next().await {
                        if let SlotUpdate::FirstShredReceived { slot, timestamp: _ } = slot_update {
                            print!("slot_subscribe_slot{}", slot);
                            datapoint_info!("slot_subscribe_slot", ("slot", slot, i64));
                            //if slot_sender.send(slot).await.is_err() {
                            //    datapoint_error!("slot_subscribe_send_error", ("errors", 1, i64));
                            //    return;
                            //}
                        }
                    }
                    slot_subscribe_disconnect_errors += 1;
                    datapoint_error!(
                        "slot_subscribe_disconnect_error",
                        ("errors", slot_subscribe_disconnect_errors, i64)
                    );
                }
                Err(e) => {
                    slot_subscribe_errors += 1;
                    datapoint_error!(
                        "slot_subscribe_error",
                        ("errors", slot_subscribe_errors, i64),
                        ("error_str", e.to_string(), String),
                    );
                }
            },
            Err(e) => {
                connect_errors += 1;
                datapoint_error!(
                    "slot_subscribe_pubsub_connect_error",
                    ("errors", connect_errors, i64),
                    ("error_str", e.to_string(), String)
                );
            }
        }
    }
}
