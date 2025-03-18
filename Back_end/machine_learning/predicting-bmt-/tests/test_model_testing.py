import sys
import os
from time import sleep
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.model_testing import load_data, reindex_features, evaluate_model, load_models

def test_end_to_end_prediction_pipeline(tmp_path):
    """
    Test the complete end-to-end prediction pipeline.
    """
    # 1. Prepare test data
    train_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [0, 1, 0, 1, 1]
    })
    test_data = pd.DataFrame({
        'feature1': [2, 3],
        'feature2': [4, 6]
    })
    
    # Save test data to CSV files
    train_csv = tmp_path / "train_data.csv"
    test_csv = tmp_path / "test_data.csv"
    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)

    # 2. Create and save a model
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    X_train = train_data[['feature1', 'feature2']]
    y_train = train_data['target']
    model.fit(X_train, y_train)

    # Create models directory
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "RandomForest_model_test.pkl"
    joblib.dump(model, model_path)

    # 3. Patch the module's file path for testing
    from src import model_testing
    original_file = model_testing.__file__
    try:
        model_testing.__file__ = str(tmp_path / "dummy_module.py")

        # 4. Execute the complete pipeline
        # Load test data
        test_df = load_data(str(test_csv.name))
        assert test_df is not None
        assert len(test_df) == 2

        # Load model
        loaded_models = load_models(str(models_dir))
        assert "RandomForest" in loaded_models

        # Prepare features
        test_features = reindex_features(test_df, loaded_models["RandomForest"])
        assert list(test_features.columns) == ['feature1', 'feature2']

        # Make predictions
        predictions = loaded_models["RandomForest"].predict(test_features)
        assert len(predictions) == 2
        assert all(isinstance(pred, (np.int64, int)) for pred in predictions)

        # Evaluate model (mock evaluation since we don't have true labels for test data)
        mock_y_test = pd.Series([0, 1])  # Mock labels for demonstration
        metrics = evaluate_model(loaded_models["RandomForest"], test_features, mock_y_test)
        
        # 5. Verify results
        assert isinstance(metrics, dict)
        assert all(key in metrics for key in ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"])
        assert all(isinstance(value, float) for value in metrics.values())
        assert all(0 <= value <= 1 for value in metrics.values())

    finally:
        model_testing.__file__ = original_file

