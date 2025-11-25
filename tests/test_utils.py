#!/usr/bin/env python3
"""
Unit Tests for Utility Functions
Author: Shashank Lodhi
Date: November 2025
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('../src')
from utils import calculate_business_metrics, create_feature_importance_df

def test_business_metrics():
    """Test business metrics calculation"""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    
    metrics = calculate_business_metrics(y_true, y_pred)
    
    assert 'true_positives' in metrics
    assert 'net_benefit' in metrics
    assert metrics['true_positives'] == 3
    assert metrics['false_positives'] == 1
    print("✅ Business metrics test passed")

def test_feature_importance():
    """Test feature importance DataFrame creation"""
    # Create simple model
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(5)]
    df = create_feature_importance_df(model, feature_names, top_n=5)
    
    assert df is not None
    assert len(df) == 5
    assert 'Feature' in df.columns
    assert 'Importance' in df.columns
    print("✅ Feature importance test passed")

if __name__ == "__main__":
    test_business_metrics()
    test_feature_importance()
    print("\n✅ All tests passed!")
