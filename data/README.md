# Data Directory

This directory contains the datasets used for the Customer Churn Prediction project.

## Directory Structure

```
data/
├── raw/               # Original unmodified datasets
│   └── Telco-Customer-Churn.csv
├── processed/         # Cleaned and processed datasets
│   ├── train_data.csv
│   └── test_data.csv
└── README.md          # This file
```

## Dataset Information

### Source
**Kaggle**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total Records | 7,043 |
| Features | 21 |
| Target Variable | Churn (Yes/No) |
| Churn Rate | 26.5% (1,869 churned) |
| Missing Values | 11 (TotalCharges column) |

### Features Description

#### Demographics
- `customerID`: Unique customer identifier
- `gender`: Male or Female
- `SeniorCitizen`: Whether customer is senior citizen (1, 0)
- `Partner`: Whether customer has partner (Yes, No)
- `Dependents`: Whether customer has dependents (Yes, No)

#### Services
- `tenure`: Number of months customer has stayed
- `PhoneService`: Whether customer has phone service (Yes, No)
- `MultipleLines`: Whether customer has multiple lines (Yes, No, No phone service)
- `InternetService`: Type of internet service (DSL, Fiber optic, No)
- `OnlineSecurity`: Whether customer has online security (Yes, No, No internet service)
- `OnlineBackup`: Whether customer has online backup (Yes, No, No internet service)
- `DeviceProtection`: Whether customer has device protection (Yes, No, No internet service)
- `TechSupport`: Whether customer has tech support (Yes, No, No internet service)
- `StreamingTV`: Whether customer has streaming TV (Yes, No, No internet service)
- `StreamingMovies`: Whether customer has streaming movies (Yes, No, No internet service)

#### Account Information
- `Contract`: Contract term (Month-to-month, One year, Two year)
- `PaperlessBilling`: Whether customer has paperless billing (Yes, No)
- `PaymentMethod`: Payment method (Electronic check, Mailed check, Bank transfer, Credit card)

#### Billing
- `MonthlyCharges`: Monthly charge amount
- `TotalCharges`: Total amount charged

#### Target
- `Churn`: Whether customer churned (Yes, No)

## How to Download

### Option 1: Manual Download
1. Visit: https://www.kaggle.com/blastchar/telco-customer-churn
2. Download `Telco-Customer-Churn.csv`
3. Place in `data/raw/` folder

### Option 2: Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d blastchar/telco-customer-churn

# Extract to data/raw/
unzip telco-customer-churn.zip -d data/raw/
```

## Data Processing

The raw data undergoes the following processing:

1. **Missing Value Treatment**: TotalCharges converted to numeric, missing values imputed with median
2. **Feature Encoding**: Categorical variables encoded (binary and one-hot)
3. **Feature Engineering**: New features created (AvgMonthlyCost, ContractBinary)
4. **Train-Test Split**: 80-20 split with stratification
5. **SMOTE Application**: Training data balanced to handle class imbalance
6. **Feature Scaling**: StandardScaler applied to numerical features

Processed datasets are saved in `data/processed/` directory.

## Data Privacy

This is a publicly available anonymized dataset. No personally identifiable information (PII) is included.

## License

The dataset is provided under Kaggle's standard data licensing terms.