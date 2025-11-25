#!/usr/bin/env python3
"""
Customer Churn Prediction - Complete ML Pipeline
Author: Shashank Lodhi
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_and_clean_data(filepath):
    """Load dataset and perform initial cleaning"""
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Handle TotalCharges (string to numeric)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)
    
    print("\n✅ Data cleaning complete!")
    return df

def perform_eda(df):
    """Create visualizations for insights"""
    print("\n📊 Performing EDA...")
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Churn distribution
    plt.figure(figsize=(8, 5))
    df['Churn'].value_counts().plot(kind='bar', color=['#21808D', '#FF5459'])
    plt.title('Churn Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('outputs/plots/churn_distribution.png', dpi=300)
    plt.close()
    
    # 2. Churn rate by Contract Type
    plt.figure(figsize=(10, 5))
    churn_contract = df.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    )
    churn_contract.plot(kind='bar', color='#32B8C6')
    plt.title('Churn Rate by Contract Type', fontsize=16, fontweight='bold')
    plt.ylabel('Churn Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/plots/churn_by_contract.png', dpi=300)
    plt.close()
    
    # 3. Churn vs Tenure
    plt.figure(figsize=(12, 6))
    df[df['Churn'] == 'Yes']['tenure'].hist(bins=30, alpha=0.7, label='Churned', color='#FF5459')
    df[df['Churn'] == 'No']['tenure'].hist(bins=30, alpha=0.7, label='Retained', color='#21808D')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Frequency')
    plt.title('Tenure Distribution by Churn Status', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/plots/tenure_churn.png', dpi=300)
    plt.close()
    
    # 4. Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/plots/correlation_heatmap.png', dpi=300)
    plt.close()
    
    print("✅ EDA complete! Plots saved to outputs/plots/")

def feature_engineering(df):
    """Create new features and encode categorical variables"""
    print("\n🔧 Engineering features...")
    
    # Create new features
    df['AvgMonthlyCost'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ContractBinary'] = df['Contract'].apply(
        lambda x: 1 if x == 'Month-to-month' else 0
    )
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode binary categorical features
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    
    # One-hot encode multi-category features
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Encode remaining categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    
    print("✅ Feature engineering complete!")
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    print("\n🤖 Training models...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, 
                                 use_label_encoder=False, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = {}
    best_model = None
    best_auc = 0
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = (name, model)
        
        print(f"  ✅ Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
    
    return results, best_model

def evaluate_best_model(best_model_name, best_model, X_test, y_test, feature_names):
    """Evaluate and visualize best model"""
    print(f"\n🏆 Best Model: {best_model_name}")
    
    y_pred = best_model.predict(X_test)
    
    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Save classification report
    with open('outputs/reports/classification_report.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('outputs/plots/confusion_matrix.png', dpi=300)
    plt.close()
    
    # Feature Importance (if applicable)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(10), importances[indices], color='#32B8C6')
        plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Top 10 Feature Importances', fontsize=16, fontweight='bold')
        plt.ylabel('Importance')
        plt.xlabel('Features')
        plt.tight_layout()
        plt.savefig('outputs/plots/feature_importance.png', dpi=300)
        plt.close()
    
    # Save best model
    joblib.dump(best_model, f'models/{best_model_name.lower().replace(" ", "_")}_model.pkl')
    print(f"\n💾 Best model saved to models/{best_model_name.lower().replace(' ', '_')}_model.pkl")

def save_results(results):
    """Save performance comparison"""
    results_df = pd.DataFrame({
        name: {
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}"
        }
        for name, metrics in results.items()
    }).T
    
    results_df.to_csv('outputs/reports/model_performance.csv')
    print("\n📁 Performance results saved to outputs/reports/model_performance.csv")
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(results_df)
    print("="*80)

def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("CUSTOMER CHURN PREDICTION - ML PIPELINE")
    print("="*80)
    
    # 1. Load and clean data
    df = load_and_clean_data('data/raw/Telco-Customer-Churn.csv')
    
    # Save original for EDA
    df_original = df.copy()
    
    # 2. Perform EDA
    perform_eda(df_original)
    
    # 3. Feature engineering
    df = feature_engineering(df)
    
    # 4. Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Handle class imbalance with SMOTE
    print("\n⚖️ Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Training set before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"Training set after SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    
    # 7. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # 8. Train models
    results, (best_model_name, best_model) = train_models(
        X_train_scaled, X_test_scaled, y_train_balanced, y_test
    )
    
    # 9. Evaluate best model
    evaluate_best_model(best_model_name, best_model, X_test_scaled, y_test, X.columns)
    
    # 10. Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("  - outputs/plots/churn_distribution.png")
    print("  - outputs/plots/churn_by_contract.png")
    print("  - outputs/plots/tenure_churn.png")
    print("  - outputs/plots/correlation_heatmap.png")
    print("  - outputs/plots/confusion_matrix.png")
    print("  - outputs/plots/feature_importance.png")
    print("  - outputs/reports/classification_report.txt")
    print("  - outputs/reports/model_performance.csv")
    print(f"  - models/{best_model_name.lower().replace(' ', '_')}_model.pkl")
    print("  - models/scaler.pkl")
    print("\n")

if __name__ == "__main__":
    main()
