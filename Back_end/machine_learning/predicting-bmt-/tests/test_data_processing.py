import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np

from src.data_processing import (
    handle_missing_values,
    remove_highly_correlated_features,
    optimize_memory,
    encode_categorical_features,
    handle_outliers,
    save_processed_data
)

pd.set_option('future.no_silent_downcasting', True)

def test_handle_missing_values():
    """
    Test that missing values and invalid entries are properly handled.
    Numeric columns are filled with the median and categorical with mode.
    """
    df = pd.DataFrame({
        'A': [1, np.nan, 3, '?', 5],
        'B': ['a', 'unknown', None, 'b', '']
    })
    df_processed = handle_missing_values(df)
    
    # For numeric column 'A', the missing values ("?", np.nan) should be replaced with the median.
    median_A = pd.to_numeric(df.replace("?", np.nan)['A'], errors='coerce').median()
    assert df_processed['A'].iloc[1] == median_A
    
    # For categorical column 'B', missing values should be replaced with the mode.
    df_B_clean = df['B'].replace(["?", "unknown", "N/A", ""], np.nan)
    mode_B = df_B_clean.mode()[0]
    # Check one of the replaced values
    assert df_processed['B'].iloc[1] == mode_B

def test_remove_highly_correlated_features():
    """
    Test that features with high correlation are removed.
    """
    # Create a DataFrame with two highly correlated features: 'A' and 'C'
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100)
    })
    df['C'] = df['A'] * 0.95 + np.random.rand(100)*0.01  # 'C' is highly correlated with 'A'
    df_reduced = remove_highly_correlated_features(df, threshold=0.9)
    
    # Either 'A' or 'C' should be dropped
    assert 'C' not in df_reduced.columns or 'A' not in df_reduced.columns

def test_optimize_memory():
    """
    Test that numeric columns are downcasted to lower-memory types.
    """
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    df_optimized = optimize_memory(df)
    
    # Check that the integer and float columns are of a numeric subtype.
    assert np.issubdtype(df_optimized['int_col'].dtype, np.integer)
    assert np.issubdtype(df_optimized['float_col'].dtype, np.floating)

def test_encode_categorical_features():
    """
    Test that categorical columns are encoded into numeric values.
    """
    df = pd.DataFrame({
        'cat': ['a', 'b', 'a', 'c'],
        'num': [1, 2, 3, 4]
    })
    df_encoded = encode_categorical_features(df)
    
    # 'cat' column should now be numeric.
    assert np.issubdtype(df_encoded['cat'].dtype, np.number)
    # Numeric column should remain unchanged.
    pd.testing.assert_series_equal(df_encoded['num'], df['num'])

def test_handle_outliers():
    """
    Test that outliers are capped using the IQR method.
    """
    # Create a DataFrame with clear outliers.
    df = pd.DataFrame({
        'A': [1, 2, 3, 100, 5],
        'B': [10, 12, 11, 13, 200]
    })
    df_out = handle_outliers(df)
    
    # For each numeric column, verify that values lie within the calculated bounds.
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        assert df_out[col].min() >= lower_bound
        assert df_out[col].max() <= upper_bound

def test_save_processed_data(tmp_path):
    """
    Test that the processed data is correctly saved to a CSV file.
    """
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    # Define a relative output path.
    output_rel_path = os.path.join("dummy_dir", "test_output.csv")
    
    # Patch __file__ in the module so that save_processed_data writes to the temporary directory.
    from src import data_processing
    original_file = data_processing.__file__
    try:
        data_processing.__file__ = str(tmp_path / "dummy_module.py")
        save_processed_data(df, output_rel_path)
        # The file should be saved at: tmp_path / dummy_dir / test_output.csv
        output_file = tmp_path / "dummy_dir" / "test_output.csv"
        assert output_file.exists(), "The processed CSV file should exist."
        df_loaded = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(df, df_loaded)
    finally:
        data_processing.__file__ = original_file

def test_end_to_end_data_processing(tmp_path):
    """
    End-to-end test for the complete data processing pipeline.
    Tests the entire chain of data transformations from raw data to processed output.
    """
    # Create a more realistic test dataset
    df = pd.DataFrame({
        'numeric_normal': [1, 2, np.nan, 4, 1000],  # Contains outlier and missing value
        'numeric_with_invalid': [1, '?', 3, 'N/A', 5],  # Contains invalid entries
        'categorical_clean': ['A', 'B', 'A', 'C', 'B'],
        'categorical_missing': ['x', 'unknown', None, '', 'x'],  # Contains missing values
        'highly_correlated_1': [1, 2, 3, 4, 5],
        'highly_correlated_2': [1.1, 2.1, 3.1, 4.1, 5.1]  # Highly correlated with highly_correlated_1
    })

    # Step 1: Handle missing values
    df_clean = handle_missing_values(df)
    
    # Verify missing values are handled
    assert not df_clean.isnull().any().any(), "Missing values should be handled"
    assert '?' not in df_clean.values, "Invalid entries should be handled"
    assert 'unknown' not in df_clean['categorical_missing'].values, "Special missing values should be handled"

    # Step 2: Handle outliers
    df_no_outliers = handle_outliers(df_clean)
    
    # Verify outliers are capped
    assert df_no_outliers['numeric_normal'].max() < 1000, "Outliers should be capped"
    
    # Step 3: Remove highly correlated features
    df_uncorrelated = remove_highly_correlated_features(df_no_outliers, threshold=0.9)
    
    # Verify correlation reduction
    assert len(df_uncorrelated.columns) < len(df_no_outliers.columns), "Highly correlated features should be removed"
    
    # Step 4: Encode categorical features
    df_encoded = encode_categorical_features(df_uncorrelated)
    
    # Verify encoding
    for col in df_encoded.columns:
        assert pd.api.types.is_numeric_dtype(df_encoded[col]), f"Column {col} should be numeric after encoding"

    # Step 5: Optimize memory
    df_optimized = optimize_memory(df_encoded)
    
    # Verify memory optimization
    memory_before = df_encoded.memory_usage().sum()
    memory_after = df_optimized.memory_usage().sum()
    assert memory_after <= memory_before, "Memory usage should be optimized"

    # Step 6: Save processed data
    output_path = "processed/test_output.csv"
    save_processed_data(df_optimized, output_path)
    
    # Verify saved data
    saved_file = tmp_path / "processed" / "test_output.csv"
    assert saved_file.parent.exists(), "Output directory should be created"
    assert saved_file.exists(), "Processed file should be saved"
    
    # Verify data integrity
    df_loaded = pd.read_csv(saved_file)
    pd.testing.assert_frame_equal(df_optimized, df_loaded)
 
