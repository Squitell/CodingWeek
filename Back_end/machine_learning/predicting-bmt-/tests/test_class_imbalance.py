import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.class_imbalance import (
    load_data,
    create_plots_dir,
    plot_class_distribution,
    balance_with_smote,
    compute_class_weights,
    split_and_save_train_test
)

@pytest.fixture
def sample_df():
    """
    Returns a small DataFrame with a binary target column 'survival_status'
    and two numeric features for testing.
    """
    data = {
        'feature1': [10, 20, 30, 40],
        'feature2': [1, 2, 3, 4],
        'survival_status': [0, 0, 1, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir(tmp_path):
    """
    Pytest's built-in fixture that provides a temporary directory unique to each test.
    """
    return tmp_path

@pytest.fixture
def real_data():
    """
    Fixture that loads a small subset of real BMT data for testing.
    """
    data = {
        'age': np.random.normal(50, 15, 100),
        'disease_risk': np.random.choice(['high', 'low', 'intermediate'], 100),
        'stem_source': np.random.choice(['BM', 'PBSC'], 100),
        'survival_status': np.random.binomial(1, 0.3, 100)  # Create imbalanced classes
    }
    df = pd.DataFrame(data)
    # Convert categorical to numeric
    df['disease_risk'] = pd.Categorical(df['disease_risk']).codes
    df['stem_source'] = pd.Categorical(df['stem_source']).codes
    return df

def test_comprehensive_end_to_end(real_data, temp_dir):
    """
    Comprehensive end-to-end test with additional validations
    """
    # 1. Initial data quality checks
    assert real_data.shape[1] >= 4, "Expected at least 4 features including target"
    assert real_data.dtypes.apply(lambda x: x.kind in 'ifc').all(), "All features should be numeric"
    
    # 2. Feature validation
    X = real_data.drop(columns=['survival_status'])
    y = real_data['survival_status']
    
    # Check feature ranges
    assert X['age'].between(0, 100).all(), "Age should be within reasonable bounds"
    assert X[['disease_risk', 'stem_source']].isin([0, 1, 2]).all().all(), "Categorical features should be encoded properly"
    
    # 3. Class imbalance validation
    initial_class_weights = compute_class_weights(y)
    class_ratios = y.value_counts(normalize=True)
    assert abs(class_ratios[0] - class_ratios[1]) > 0.2, "Expected significant class imbalance"
    
    # 4. SMOTE balancing validation with cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_fold, y_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_balanced, y_balanced = balance_with_smote(X_fold, y_fold, random_state=42)
        
        # Check balancing results
        fold_counts = pd.Series(y_balanced).value_counts()
        assert abs(fold_counts[0] - fold_counts[1]) <= 1, f"Fold {fold} is not properly balanced"
        assert not X_balanced.isna().any().any(), f"Found NaN values in fold {fold}"
        
    # 5. Final train-test split validation
    with patch('src.class_imbalance.__file__', os.path.join(temp_dir, 'class_imbalance.py')):
        processed_dir = os.path.join(temp_dir, "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        X_final, y_final = balance_with_smote(X, y, random_state=42)
        split_and_save_train_test(X_final, y_final, test_size=0.2, random_state=42)
        
        # Load and validate split files
        train_df = pd.read_csv(os.path.join(processed_dir, "bmt_train.csv"))
        test_df = pd.read_csv(os.path.join(processed_dir, "bmt_test.csv"))
        
        # Additional validations on split data
        for name, df in [("train", train_df), ("test", test_df)]:
            # Check class distribution
            split_ratios = df['survival_status'].value_counts(normalize=True)
            assert abs(split_ratios[0] - split_ratios[1]) < 0.1, f"Imbalanced classes in {name} set"
            
            # Check feature correlations
            correlations = df.corr()
            assert not (correlations.abs() > 0.95).any().any(), f"Found highly correlated features in {name} set"
            
            # Check for data leakage
            common_indices = set(train_df.index) & set(test_df.index)
            assert len(common_indices) == 0, "Found data leakage between train and test sets"

