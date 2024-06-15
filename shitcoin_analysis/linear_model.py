import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import polars as pl
def train_and_evaluate(X_train, X_test, y_train, y_test, alpha=1.0):
    # Train Ridge Regression model (L2 regularization)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    ridge_mse = mean_squared_error(y_test, y_pred_ridge)
    ridge_r2 = r2_score(y_test, y_pred_ridge)
    print(f"Ridge Regression: MSE = {ridge_mse}, R2 = {ridge_r2}")

    # Train Lasso Regression model (L1 regularization)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    lasso_mse = mean_squared_error(y_test, y_pred_lasso)
    lasso_r2 = r2_score(y_test, y_pred_lasso)
    print(f"Lasso Regression: MSE = {lasso_mse}, R2 = {lasso_r2}")

    return ridge, lasso

# Load the processed features and targets
def load_data(feature_file, target_file):
    features = pl.read_csv(feature_file)
    targets = pl.read_csv(target_file)
    return features, targets

# Preprocess data and get feature matrix (X) and target vector (y)
def preprocess_data(features_df, targets_df):
    X = features_df.to_numpy()
    y = targets_df.to_numpy()
    return X, y

if __name__ == "__main__":
    feature_file = 'path_to_combined_features.csv'
    target_file = 'path_to_combined_targets.csv'
    
    features_df, targets_df = load_data(feature_file, target_file)
    X, y = preprocess_data(features_df, targets_df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the models with L1 and L2 regularization
    train_and_evaluate(X_train, X_test, y_train, y_test, alpha=1.0)
