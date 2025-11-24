# Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv       # Original dataset from Kaggle
│   │   └── README.md                      # Data documentation
│   ├── processed/
│   │   ├── train_data.csv                 # Training set (80%)
│   │   ├── test_data.csv                  # Testing set (20%)
│   │   └── feature_engineered.csv         # With derived features
│   └── README.md                          # Data folder documentation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Initial EDA and statistics
│   ├── 02_feature_engineering.ipynb       # Feature creation and encoding
│   ├── 03_model_training.ipynb            # Model experiments and tuning
│   ├── 04_model_evaluation.ipynb          # Performance analysis and comparison
│   └── 05_business_insights.ipynb         # Business recommendations
│
├── src/
│   ├── __init__.py
│   ├── churn_prediction_pipeline.py       # Main ML pipeline script
│   ├── streamlit_app.py                   # Interactive Streamlit dashboard
│   ├── data_preprocessing.py              # Data cleaning functions
│   ├── feature_engineering.py             # Feature creation functions
│   ├── model_training.py                  # Model training logic
│   ├── model_evaluation.py                # Evaluation metrics
│   ├── visualization.py                   # Plotting functions
│   └── utils.py                           # Helper utilities
│
├── models/
│   ├── xgboost_model.pkl                  # Trained XGBoost model (best)
│   ├── random_forest_model.pkl            # Trained Random Forest
│   ├── logistic_regression_model.pkl      # Trained Logistic Regression
│   ├── gradient_boosting_model.pkl        # Trained Gradient Boosting
│   ├── svm_model.pkl                      # Trained SVM
│   ├── scaler.pkl                         # StandardScaler object
│   ├── label_encoders.pkl                 # Label encoders for categorical features
│   ├── model_config.json                  # Hyperparameters for all models
│   └── README.md                          # Model documentation
│
├── outputs/
│   ├── plots/
│   │   ├── churn_distribution.png             # Overall churn rate pie chart
│   │   ├── churn_by_contract.png              # Churn vs contract type bar chart
│   │   ├── churn_by_tenure.png                # Churn vs tenure histogram
│   │   ├── churn_by_payment.png               # Churn vs payment method
│   │   ├── tenure_churn.png                   # Tenure distribution by churn status
│   │   ├── monthly_charges_dist.png           # Monthly charges distribution
│   │   ├── correlation_heatmap.png            # Feature correlation matrix
│   │   ├── confusion_matrix.png               # XGBoost confusion matrix
│   │   ├── roc_curve.png                      # ROC curve comparison
│   │   ├── feature_importance.png             # Top 10 features by importance
│   │   ├── precision_recall_curve.png         # Precision-recall trade-off
│   │   └── model_comparison.png               # Accuracy comparison bar chart
│   └── reports/
│       ├── model_performance.txt              # Detailed performance metrics
│       ├── classification_report.txt          # Precision/Recall/F1 report
│       ├── business_insights.txt              # Key findings and recommendations
│       └── data_summary_statistics.txt        # Dataset statistics
│
├── dashboard/
│   ├── index.html                         # Standalone HTML dashboard
│   ├── assets/
│   │   ├── style.css                      # Custom CSS (if separated)
│   │   └── script.js                      # Custom JS (if separated)
│   └── README.md                          # Dashboard documentation
│
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py         # Unit tests for data cleaning
│   ├── test_feature_engineering.py        # Unit tests for feature creation
│   ├── test_model_predictions.py          # Unit tests for model predictions
│   └── test_utils.py                      # Unit tests for utilities
│
├── docs/
│   ├── project_report.pdf                 # Comprehensive project report
│   ├── presentation.pptx                  # Project presentation slides
│   ├── methodology.md                     # Detailed methodology
│   └── references.md                      # Research papers and resources
│
├── config/
│   ├── model_config.yaml                  # Model hyperparameters
│   ├── data_config.yaml                   # Data processing configs
│   └── paths.yaml                         # File paths configuration
│
├── scripts/
│   ├── download_data.sh                   # Script to download Kaggle dataset
│   ├── setup_environment.sh               # Environment setup script
│   └── run_pipeline.sh                    # Execute full pipeline
│
├── .gitignore                             # Git ignore patterns
├── requirements.txt                       # Python dependencies
├── setup.py                               # Package setup file
├── LICENSE                                # MIT License
├── README.md                              # Main project documentation
├── STRUCTURE.md                           # This file - project structure
└── CONTRIBUTING.md                        # Contribution guidelines
```

## File Descriptions

### Data Files (`data/`)
- **raw/**: Unmodified datasets from Kaggle
- **processed/**: Cleaned and feature-engineered datasets ready for modeling

### Notebooks (`notebooks/`)
- Jupyter notebooks for exploratory analysis and experimentation
- Numbered for sequential workflow

### Source Code (`src/`)
- Production-ready Python modules
- Main pipeline script for end-to-end execution
- Streamlit dashboard for interactive predictions

### Models (`models/`)
- Serialized trained models (.pkl files)
- Preprocessing objects (scalers, encoders)
- Configuration files with hyperparameters

### Outputs (`outputs/`)
- **plots/**: All visualizations (PNG format, 300 DPI)
- **reports/**: Text-based performance metrics and insights

### Dashboard (`dashboard/`)
- Standalone HTML dashboard for deployment
- No server required - runs in browser

### Tests (`tests/`)
- Unit tests for all modules
- Run with: `pytest tests/`

### Documentation (`docs/`)
- Comprehensive project documentation
- Presentation materials for showcasing

### Configuration (`config/`)
- YAML files for centralized configuration
- Easy parameter tuning without code changes

### Scripts (`scripts/`)
- Shell scripts for automation
- Dataset download and environment setup

## How Files Interact

```
Kaggle Dataset → data/raw/ → data_preprocessing.py → data/processed/
                                       ↓
                            feature_engineering.py
                                       ↓
                            model_training.py → models/
                                       ↓
                            model_evaluation.py → outputs/
                                       ↓
                            streamlit_app.py / dashboard/
```

## Adding New Components

### To add a new model:
1. Create training logic in `src/model_training.py`
2. Save model to `models/`
3. Add evaluation in `src/model_evaluation.py`
4. Update dashboard to include new model

### To add new visualizations:
1. Create plotting function in `src/visualization.py`
2. Save plots to `outputs/plots/`
3. Reference in notebooks or dashboard

### To add new features:
1. Implement in `src/feature_engineering.py`
2. Update feature list in documentation
3. Retrain models with new features
4. Compare performance

## File Naming Conventions

- **Python modules**: `snake_case.py`
- **Notebooks**: `##_description.ipynb` (numbered)
- **Data files**: `descriptive_name.csv`
- **Model files**: `model_name_model.pkl`
- **Plot files**: `descriptive_name.png`
- **Config files**: `purpose_config.yaml`

## Git Workflow

1. All data files (`.csv`, models `.pkl`) are in `.gitignore`
2. Only code, configs, and documentation are version controlled
3. Use Git LFS for large files if needed
4. Keep commits focused and atomic

---

*This structure follows industry best practices for ML projects and is designed for scalability and collaboration.*