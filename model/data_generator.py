# model/data_generator.py
"""
Utility script to generate synthetic data for testing the ML pipeline.
This can be useful when you want to test your MLOps setup without real data.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_classification_data(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    output_file='synthetic_data.csv'
):
    """Generate synthetic classification data and save it to a CSV file."""
    
    print(f"Generating synthetic dataset with {n_samples} samples and {n_features} features...")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),  # 80% of features are informative
        n_redundant=int(n_features * 0.1),    # 10% are redundant
        n_classes=n_classes,
        random_state=42
    )
    
    # Create feature names and convert to dataframe
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    
    # Print data summary
    print("\nData summary:")
    print(f"Shape: {df.shape}")
    print(f"Class distribution:\n{df['target'].value_counts()}")
    
    return df

if __name__ == "__main__":
    generate_classification_data()
