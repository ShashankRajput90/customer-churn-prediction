#!/usr/bin/env python3
"""
Utility Functions for Customer Churn Prediction
Author: Shashank Lodhi
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import joblib

def save_model(model, filename):
    """
    Save trained model to file
    
    Args:
        model: Trained sklearn/xgboost model
        filename: Path to save model
    """
    joblib.dump(model, filename)
    print(f"✅ Model saved to {filename}")

def load_model(filename):
    """
    Load trained model from file
    
    Args:
        filename: Path to model file
    
    Returns:
        Loaded model
    """
    model = joblib.load(filename)
    print(f"✅ Model loaded from {filename}")
    return model

def plot_roc_curves(models_dict, X_test, y_test, save_path=None):
    """
    Plot ROC curves for multiple models
    
    Args:
        models_dict: Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test labels
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ ROC curves saved to {save_path}")
    
    plt.show()

def calculate_business_metrics(y_true, y_pred, customer_value=2000, campaign_cost=50):
    """
    Calculate business impact metrics
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        customer_value: Average customer lifetime value
        campaign_cost: Cost per retention campaign
    
    Returns:
        Dictionary of business metrics
    """
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate financial impact
    revenue_saved = tp * customer_value  # Successfully retained customers
    campaign_costs = (tp + fp) * campaign_cost  # Cost of targeting
    missed_revenue = fn * customer_value  # Customers we missed
    
    net_benefit = revenue_saved - campaign_costs
    roi = (net_benefit / campaign_costs) * 100 if campaign_costs > 0 else 0
    
    metrics = {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'revenue_saved': revenue_saved,
        'campaign_costs': campaign_costs,
        'missed_revenue': missed_revenue,
        'net_benefit': net_benefit,
        'roi_percentage': roi
    }
    
    return metrics

def print_business_report(metrics):
    """
    Print formatted business impact report
    
    Args:
        metrics: Dictionary from calculate_business_metrics
    """
    print("\n" + "="*60)
    print("BUSINESS IMPACT REPORT")
    print("="*60)
    print(f"\n📊 Prediction Breakdown:")
    print(f"  True Positives (Correctly identified churners): {metrics['true_positives']}")
    print(f"  False Positives (False alarms): {metrics['false_positives']}")
    print(f"  False Negatives (Missed churners): {metrics['false_negatives']}")
    print(f"  True Negatives (Correctly identified retained): {metrics['true_negatives']}")
    
    print(f"\n💰 Financial Impact:")
    print(f"  Revenue Saved: ${metrics['revenue_saved']:,.0f}")
    print(f"  Campaign Costs: ${metrics['campaign_costs']:,.0f}")
    print(f"  Missed Revenue: ${metrics['missed_revenue']:,.0f}")
    print(f"  Net Benefit: ${metrics['net_benefit']:,.0f}")
    print(f"  ROI: {metrics['roi_percentage']:.1f}%")
    print("="*60 + "\n")

def create_feature_importance_df(model, feature_names, top_n=10):
    """
    Create DataFrame of feature importances
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame with features and importances
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Model does not have feature_importances_ attribute")
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices],
        'Importance_Normalized': (importances[indices] / importances.max()) * 100
    })
    
    return df

def export_predictions(y_true, y_pred, y_pred_proba, customer_ids=None, filename='predictions.csv'):
    """
    Export predictions to CSV file
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        customer_ids: Optional customer IDs
        filename: Output filename
    """
    df = pd.DataFrame({
        'CustomerID': customer_ids if customer_ids is not None else range(len(y_true)),
        'Actual_Churn': y_true,
        'Predicted_Churn': y_pred,
        'Churn_Probability': y_pred_proba,
        'Risk_Category': pd.cut(
            y_pred_proba,
            bins=[0, 0.35, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    })
    
    df.to_csv(filename, index=False)
    print(f"✅ Predictions exported to {filename}")
    
    return df

def get_risk_summary(y_pred_proba):
    """
    Generate summary of risk distribution
    
    Args:
        y_pred_proba: Array of churn probabilities
    
    Returns:
        Dictionary with risk category counts
    """
    risk_categories = pd.cut(
        y_pred_proba,
        bins=[0, 0.35, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    summary = risk_categories.value_counts().to_dict()
    
    print("\n📊 Risk Distribution:")
    for category, count in summary.items():
        percentage = (count / len(y_pred_proba)) * 100
        print(f"  {category} Risk: {count} ({percentage:.1f}%)")
    
    return summary
