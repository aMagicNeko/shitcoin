import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from feature_reader import process_files_randomly, preprocess_data, read_and_process_polars

def split_data(features, target, test_size=0.2, random_state=42):
    return train_test_split(features, target, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, n_estimators=100, random_state=42, class_weight='balanced'):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight=class_weight)
    model.fit(X_train, y_train)
    return model

def save_model(model, path='random_forest_model.joblib'):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path='random_forest_model.joblib'):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, precision, recall, f1, roc_auc

def print_evaluation_metrics(accuracy, precision, recall, f1, roc_auc):
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

def feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature importances:")
    for i in range(len(feature_names)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def save_model_as_onnx(model, path='random_forest_model.onnx', initial_types=None, target_opset=9):
    # Convert the sklearn model to ONNX format with specified opset version
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=target_opset)
    # Save the ONNX model to a file
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model saved to {path} with opset version {target_opset}")

if __name__ == "__main__":
    feature, target = process_files_randomly('processed_data', 6000)
    feature, target = preprocess_data(feature, target, feature.columns, target.columns, True, 'random_forest_preprocess.json')
    feature = feature.drop('slot')
    # 分割数据集
    X_train, X_test, y_train, y_test = split_data(feature, target['token0_drop_10%_15slots'])

    # 训练模型
    model = train_model(X_train, y_train)

    # 保存模型为 joblib 格式
    save_model(model)

    # 加载模型
    model = load_model()

    # 评估模型
    accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)
    print_evaluation_metrics(accuracy, precision, recall, f1, roc_auc)

    # 打印特征重要性
    feature_importance(model, feature.columns)

    # 保存模型为 ONNX
    initial_types = [('float_input', FloatTensorType([None, feature.shape[1]]))]
    save_model_as_onnx(model, initial_types=initial_types)
