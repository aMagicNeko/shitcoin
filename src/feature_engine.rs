use std::collections::{HashMap, VecDeque, BTreeSet};
use std::f64::consts::E;
use solana_sdk::pubkey::Pubkey;
use std::cmp::Ordering;
use crate::strategy::Step;
const FLOAT_TOLERANCE: f32 = 1e-4;

#[derive(Debug, Clone)]
struct Holding {
    address: Pubkey,
    amount: f32,
}

impl Eq for Holding {}

impl PartialEq for Holding {
    fn eq(&self, other: &Self) -> bool {
        (self.amount - other.amount).abs() < FLOAT_TOLERANCE && self.address == other.address
    }
}

impl Ord for Holding {
    fn cmp(&self, other: &Self) -> Ordering {
        if (self.amount - other.amount).abs() < FLOAT_TOLERANCE {
            self.address.cmp(&other.address)
        } else {
            self.amount.partial_cmp(&other.amount).unwrap_or(Ordering::Equal).reverse()
        }
    }
}

impl PartialOrd for Holding {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct SlotData {
    token0_value: f32,
    total_inflow: f32,
    total_outflow: f32,
    top5_holding: f32,
    top10_holding: f32,
    max_holding: f32,
    holding_entropy: f32,
    num_addresses: usize,
    negative_holdings: f32,
}

#[derive(Debug)]
pub struct FeatureExtractor {
    window_size: usize,
    slot_data: VecDeque<SlotData>,
    address_holdings: HashMap<Pubkey, f32>,
    ordered_positive_holdings: BTreeSet<Holding>, // 正持有量的有序集合
    total_token1: f32, // 总的token0
    total_negative_holdings: f32, // 负持有量的总和
    current_slot: u64,
    current_inflow: f32,
    current_outflow: f32,
    current_token0: f32,
    current_token1: f32,
    open_token0: f32,
    open_token1: f32,
    open_liquidity: f32,
    start_slot: u64,
    inslot_order: HashMap<Pubkey, (f32, f32)>,
    prev_slot: u64,
}

impl FeatureExtractor {
    pub fn new(window_size: usize, start_slot: u64, open_token0: f32, open_token1: f32) -> Self {
        Self {
            window_size,
            slot_data: VecDeque::with_capacity(window_size),
            address_holdings: HashMap::new(),
            ordered_positive_holdings: BTreeSet::new(), // 初始化正持有量的有序集合
            total_token1: 0.0, // 初始化总的token0
            total_negative_holdings: 0.0, // 初始化负持有量的总和
            current_slot: start_slot,
            current_inflow: 0.0,
            current_outflow: 0.0,
            current_token0: open_token0,
            current_token1: open_token1,
            open_token0,
            open_token1,
            open_liquidity: open_token0 * open_token1,
            start_slot,
            inslot_order: HashMap::new(),
            prev_slot: start_slot,
        }
    }

    pub fn on_slot(&mut self, slot: u64) {
        if self.current_slot != slot {
            let mut slot_inflow = 0.0;
            let mut slot_outflow = 0.0;
            for (from_addr, delta) in &self.inslot_order {
                let holding = self.address_holdings.entry(*from_addr).or_insert(0.0);
                if *holding > 0.0 {
                    self.ordered_positive_holdings.remove(&Holding { address: *from_addr, amount: *holding }); // 移除旧持有量
                    self.total_token1 -= *holding;
                } else {
                    self.total_negative_holdings -= *holding;
                }
                *holding -= delta.1;
                if *holding > 0.0 {
                    self.ordered_positive_holdings.insert(Holding { address: *from_addr, amount: *holding }); // 插入新持有量
                    self.total_token1 += *holding;
                } else {
                    self.total_negative_holdings += *holding;
                }
                if delta.0 > 0.0 {
                    slot_inflow += delta.0;
                } else {
                    slot_outflow += delta.0;
                }
            }
            self.current_inflow += slot_inflow;
            self.current_outflow += slot_outflow;
            self.inslot_order.clear();
            for _ in self.current_slot..slot {
                self.append_current_slot_data();
            }
        }
        self.prev_slot = self.current_slot;
        self.current_slot = slot;
    }

    pub fn update(&mut self, step: &Step) {
        let Step{slot, token0, token1, delta0, delta1, from} = step;
        if self.current_slot != *slot {
            self.on_slot(*slot);
        }

        self.current_slot = *slot;
        self.current_token0 = (token0 + delta0) as f32;
        self.current_token1 = (token1 + delta1) as f32;

        let entry = self.inslot_order.entry(*from).or_insert((0.0, 0.0));
        entry.0 += *delta0 as f32;
        entry.1 += *delta1 as f32;
    }

    fn append_current_slot_data(&mut self) {
        let total_token1 = self.total_token1;
        let negative_holdings = self.total_negative_holdings;
    
        // 计算top5持有量比例
        let top5_holding = if total_token1 > 0.0 {
            self.ordered_positive_holdings.iter().take(5).map(|h| h.amount).sum::<f32>() / total_token1
        } else {
            1.0
        };
    
        // 计算top10持有量比例
        let top10_holding = if total_token1 > 0.0 {
            self.ordered_positive_holdings.iter().take(10).map(|h| h.amount).sum::<f32>() / total_token1
        } else {
            1.0
        };
    
        // 计算最大持有量比例
        let max_holding = if let Some(max) = self.ordered_positive_holdings.iter().next() {
            max.amount / total_token1
        } else {
            1.0
        };
    
        // 计算持有量分布
        let holding_distribution: Vec<f32> = if total_token1 > 0.0 {
            self.ordered_positive_holdings.iter().map(|h| h.amount / total_token1).collect()
        } else {
            vec![1.0]
        };
        let holding_entropy = entropy(&holding_distribution);
    
        // 创建新的SlotData并添加到slot_data队列中
        let slot_data = SlotData {
            token0_value: self.current_token0,
            total_inflow: self.current_inflow,
            total_outflow: self.current_outflow,
            top5_holding,
            top10_holding,
            max_holding,
            holding_entropy,
            num_addresses: self.ordered_positive_holdings.len(),
            negative_holdings,
        };
    
        self.slot_data.push_back(slot_data);
        if self.slot_data.len() > self.window_size {
            self.slot_data.pop_front();
        }
    }

    pub fn compute_features(&self) -> Vec<f32> {
        let mut features = Vec::new();
        let slot_windows = [15, 30, 60, 120, 240, 480, 960, 1920, 3840, 7680];
    
        features.push(self.prev_slot as f32); // slot: 0
        features.push(self.current_inflow); // cumulative_inflow: 1
        features.push(self.current_outflow); // cumulative_outflow: 2
        features.push(self.open_token0); // open_token0: 3
        features.push(self.open_token1); // open_token1: 4
        features.push(self.open_liquidity); // open_liquidity: 5
        features.push(self.current_token0); // current_token0: 6
        features.push(self.current_token1); // current_token1: 7
    
        let current_liquidity = self.current_token0 * self.current_token1;
        features.push(current_liquidity / self.open_liquidity); // current_liquidity_ratio: 8
        features.push(self.current_token0 / self.open_token0); // current_to_open_token0_ratio: 9
        features.push((self.prev_slot - self.start_slot) as f32); // slot_elapse: 10
    
        for &window in &slot_windows {
            let window_slot = if window < self.slot_data.len() {
                self.slot_data[self.slot_data.len() - window - 1].clone()
            }
            else {
                SlotData {
                    token0_value: self.open_token0,
                    total_inflow: 0.0,
                    total_outflow: 0.0,
                    top5_holding: 1.0,
                    top10_holding: 1.0,
                    max_holding: 1.0,
                    holding_entropy: 0.0,
                    num_addresses: 0,
                    negative_holdings: 0.0,
                }
            };
            features.push(window_slot.token0_value); // token0_value_{window}slots: 11, 17, 23, 29, 35, 41, 47, 53, 59, 65
            features.push(window_slot.token0_value / self.open_token0); // token0_relative_value_{window}slots: 12, 18, 24, 30, 36, 42, 48, 54, 60, 66
            features.push(self.current_token0 - window_slot.token0_value); // token0_diff_value_{window}slots: 13, 19, 25, 31, 37, 43, 49, 55, 61, 67
            features.push((self.current_token0 - window_slot.token0_value) / self.open_token0); // token0_relative_diff_value_{window}slots: 14, 20, 26, 32, 38, 44, 50, 56, 62, 68

            let window_inflow = self.current_inflow - window_slot.total_inflow;
            let window_outflow = self.current_outflow - window_slot.total_outflow;
            features.push(window_inflow); // inflow_{window}slots: 15, 21, 27, 33, 39, 45, 51, 57, 63, 69
            features.push(window_outflow); // outflow_{window}slots: 16, 22, 28, 34, 40, 46, 52, 58, 64, 70
        }

        for (i, windows) in slot_windows[..9].iter().enumerate() {
            let inflow_diff = features[15 + (i + 1) * 6] - features[15 + i * 6];            
            features.push(inflow_diff); // inflow_diff_{window}slots: 71, 73, 75, 77, 79, 81, 83, 85, 87

            let outflow_diff = features[16 + (i + 1) * 6] - features[16 + i * 6];
            features.push(outflow_diff); // outflow_diff_{window}slots: 72, 74, 76, 78, 80, 82, 84, 86, 88
        }
           
        let (negative_holdings, num_addresses, max_address_holding, top_5_address_holding, top_10_address_holding, holding_entropy) = if let Some(slot_data) = self.slot_data.back() {
            (slot_data.negative_holdings, slot_data.num_addresses as f32, slot_data.max_holding, slot_data.top5_holding, slot_data.top10_holding, slot_data.holding_entropy)
        }
        else {
            (0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
        };
        features.push(negative_holdings); // negative_holdings: 89
        features.push(num_addresses); // num_addresses: 90
        features.push(max_address_holding); // max_address_holding: 91
        features.push(top_5_address_holding); // top_5_address_holding: 92
        features.push(top_10_address_holding); // top_10_address_holding: 93
        features.push(holding_entropy); // holding_entropy: 94
    
        for &window in &slot_windows {
            let (window_negative_holdings, window_num_addresses, window_max_address_holding, window_top_5_address_holding, window_top_10_address_holding, window_holding_entropy) = if window < self.slot_data.len() {
                let slot_data = &self.slot_data[self.slot_data.len() - window - 1];
                (slot_data.negative_holdings, slot_data.num_addresses as f32, slot_data.max_holding, slot_data.top5_holding, slot_data.top10_holding, slot_data.holding_entropy)
            }
            else {
                (0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
            };
            features.push(top_5_address_holding - window_top_5_address_holding); // top_5_address_holding_diff_{window}slots: 95
            features.push(top_10_address_holding - window_top_10_address_holding); // top_10_address_holding_diff_{window}slots: 96
            features.push(max_address_holding - window_max_address_holding); // max_address_holding_diff_{window}slots: 97
            features.push(num_addresses - window_num_addresses); // num_addresses_diff_{window}slots: 98
            features.push(negative_holdings - window_negative_holdings); // negative_holdings_diff_{window}slots: 99
            features.push(holding_entropy - window_holding_entropy); // holding_entropy_diff_{window}slots: 100
        }
        features
    }
}

// Helper function to compute entropy
fn entropy(probs: &[f32]) -> f32 {
    if probs.is_empty() || probs.iter().all(|&p| p == 0.0) {
        return 0.0;
    }
    probs.iter().fold(0.0, |acc, &p| if p > 0.0 { acc - p * p.log(E as f32)} else { acc })
}
