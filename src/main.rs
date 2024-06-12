mod subscription;
mod router;
mod raydium_amm;
mod strategy;
mod transaction_executor;
use log::info;
use env_logger;
use solana_sdk::signer::{keypair::{read_keypair_file, Keypair}, Signer};
use hex::FromHex;
use transaction_executor::{start_get_block_hash_loop};
#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Logging is initialized.");
    //info!("pubkey: {}", KEYPAIR.pubkey());
    subscription::start_slot_subscription();
    start_get_block_hash_loop();
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    //init_token_account().await.unwrap();
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    // 启动订阅任务
    subscription::start_subscription(tx);
    // 启动处理任务
    router::start_router(rx);

    // 主线程继续运行
    info!("Main thread is running...");
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
    }
    
}
