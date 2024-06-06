mod subscription;
mod router;
mod raydium_amm;
mod strategy;
use log::info;
use env_logger;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Logging is initialized.");

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
