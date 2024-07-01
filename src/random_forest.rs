use ort::{GraphOptimizationLevel, Session, DynValue, Map, Allocator, ValueRef, DynMap, MapValueType, DynSequenceValueType, DynValueTypeMarker};
use ndarray::{ArrayD, IxDyn, Array1, Array2};
use once_cell::sync::Lazy;
use log::info;
use anyhow::anyhow;
use anyhow::Error;
use std::collections::HashMap;
use serde::Deserialize;
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;
#[derive(Deserialize, Debug)]
struct DataPoint {
    mean: f32,
    std: f32,
}

static RANDOM_FOREST: Lazy<Session> = Lazy::new(|| {
    Session::builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .unwrap()
                .with_intra_threads(4)
                .unwrap()
                .commit_from_file("random_forest_model.onnx")
                .unwrap()
});

static GLOBAL_DATA: Lazy<(Vec<f32>, Vec<f32>)> = Lazy::new(|| {
    let file = File::open("random_forest_preprocess.json").expect("Failed to open file");
    let reader = BufReader::new(file);
    let json_data: Value = serde_json::from_reader(reader).expect("Failed to read JSON");

    let mut means = Vec::new();
    let mut stds = Vec::new();

    if let Value::Object(map) = json_data {
        for (_key, value) in map {
            if let Ok(data_point) = serde_json::from_value::<DataPoint>(value) {
                means.push(data_point.mean);
                stds.push(data_point.std);
            }
        }
    }
    (means, stds)
});


pub fn print_model_dim() {
    info!("random forest model input:{:?}, output{:?}", RANDOM_FOREST.inputs, RANDOM_FOREST.outputs);
}

pub fn predict(features: Vec<f32>) -> Result<f32, Error> {
    let (means, stds) = &*GLOBAL_DATA;
    let standardized_features: Vec<f32> = features.iter().enumerate()
        .map(|(i, &x)| ((x - means[i]) / stds[i]) )
        .collect();
    let standardized_features = &standardized_features[1..];
    let input_data = Array2::from_shape_vec((1, 154), standardized_features.to_vec()).expect("Failed to create input array");
    let input_tensor = ort::inputs![input_data].expect("Failed to create input tensor");

    let outputs = RANDOM_FOREST.run(input_tensor).expect("Failed to run prediction");
    info!("predict: {:?}", outputs);
    let allocator = Allocator::default();
    let predictions_ref = outputs["output_probability"].try_extract_sequence::<DynValueTypeMarker>(&allocator)?;

    for pred in predictions_ref {
        let map: HashMap<i64, f32> = pred.try_extract_map(&allocator)?;
        let probability = map.get(&1).expect("map error");
        return Ok(*probability);
    }
    Err(anyhow!("no resut"))
}