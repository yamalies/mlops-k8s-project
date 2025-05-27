import os
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load dataset from CSV file."""
    logger.info(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df, target_column='target'):
    """Preprocess the dataset."""
    logger.info("Preprocessing data")
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train, hyperparams=None):
    """Train a RandomForest model with optional hyperparameters."""
    logger.info("Training model")
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    logger.info("Evaluating model")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{report}")
    
    return accuracy, report

def save_model(model, scaler, output_dir="/models"):
    """Save model artifacts."""
    logger.info(f"Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and scaler
    model_path = os.path.join(output_dir, "model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save model metadata
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        f.write(f"Model type: RandomForestClassifier\n")
        f.write(f"Training timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Feature count: {model.n_features_in_}\n")
        f.write(f"Class count: {len(model.classes_)}\n")
    
    logger.info("Model saved successfully")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--data-path", type=str, default="/data/train.csv", help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="/models", help="Directory to save the model")
    parser.add_argument("--retraining", action="store_true", help="Whether this is a retraining job")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()

def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Configure MLflow if tracking URI is provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    
    experiment_name = "model-retraining" if args.retraining else "model-training"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("retraining", args.retraining)
        
        try:
            # Load and preprocess data
            df = load_data(args.data_path)
            X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
            
            # Define hyperparameters
            hyperparams = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
            
            # Log hyperparameters
            mlflow.log_params(hyperparams)
            
            # Train model
            model = train_model(X_train, y_train, hyperparams)
            
            # Evaluate model
            accuracy, _ = evaluate_model(model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # Save model
            save_model(model, scaler, args.output_dir)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            mlflow.log_param("error", str(e))
            raise

if __name__ == "__main__":
    main()
