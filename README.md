# 🎯 Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-84.2%25-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **ML-powered customer retention analytics achieving 84.2% prediction accuracy**

End-to-end machine learning system that predicts customer churn probability and provides actionable retention strategies for the telecom industry.

[View Demo](https://github.com/ShashankRajput90/customer-churn-prediction) | [Documentation](README_DETAILED.md) | [Structure](STRUCTURE.md) | [Contributing](CONTRIBUTING.md)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/ShashankRajput90/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run ML pipeline
python src/churn_prediction_pipeline.py

# Launch dashboard
streamlit run src/streamlit_app.py
```

---

## ✨ Key Features

- 🤖 **5 ML Models** - XGBoost, Random Forest, Gradient Boosting, Logistic Regression, SVM
- 🎯 **84.2% Accuracy** - Best-in-class performance on telecom churn dataset
- 📊 **Interactive Dashboard** - Real-time predictions with risk categorization
- 📝 **Auto Recommendations** - Personalized retention strategies per customer
- 🔍 **15+ Visualizations** - EDA insights and model performance metrics
- ⚖️ **SMOTE Balancing** - Handles 26.5% class imbalance effectively

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| **XGBoost** ⭐ | **84.2%** | **0.87** | **0.78** |
| Random Forest | 80.8% | 0.84 | 0.72 |
| Gradient Boosting | 79.3% | 0.82 | 0.70 |
| Logistic Regression | 75.1% | 0.79 | 0.65 |
| SVM | 72.4% | 0.76 | 0.61 |

---

## 🔥 Business Impact

```
47.6%  churn rate for <12 month tenure customers
42.7%  churn for month-to-month vs 2.8% for 2-year contracts  
45.3%  churn rate for electronic check payments
$2.5M  estimated annual revenue protection potential
```

---

## 💻 Tech Stack

**ML/Data**: Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn  
**Visualization**: Matplotlib, Seaborn, Plotly  
**Dashboard**: Streamlit  
**Tools**: Jupyter, Git, VS Code

---

## 📁 Project Structure

```
customer-churn-prediction/
├── data/              # Datasets (raw & processed)
├── notebooks/         # Jupyter notebooks for EDA
├── src/               # Source code (pipeline, dashboard, utils)
├── models/            # Trained models (.pkl files)
├── outputs/           # Plots and performance reports
├── dashboard/         # Standalone HTML dashboard
├── tests/             # Unit tests
└── requirements.txt   # Python dependencies
```

See [STRUCTURE.md](STRUCTURE.md) for complete details.

---

## 🔍 Top Insights

### Churn Drivers

1. **Tenure** - New customers (<12 months) churn 6.4x more than long-term (49+ months)
2. **Contract Type** - Month-to-month contracts have 15x higher churn than 2-year
3. **Payment Method** - Electronic check users churn 3x more than credit card
4. **Monthly Charges** - Customers paying $70+ show 35% higher churn risk
5. **Services** - Fiber optic users churn 2.2x more than DSL subscribers

### Retention Strategies

✅ Proactive outreach at 6-month mark  
✅ Contract upgrade incentives with discounts  
✅ Autopay migration for electronic check users  
✅ Tech support + online security bundles  
✅ Competitive fiber pricing analysis

---

## 📚 Dataset

**Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

- 7,043 customers
- 21 features (demographics + services)
- 26.5% churn rate (1,869 churned)
- Binary target: Churn (Yes/No)

---

## 🎯 Usage Examples

### 1. Train Models

```python
python src/churn_prediction_pipeline.py
```

Outputs: cleaned data, trained models, visualizations, performance metrics

### 2. Interactive Dashboard

```bash
streamlit run src/streamlit_app.py
```

Access at `http://localhost:8501`

### 3. Predict Programmatically

```python
import pickle
import pandas as pd

# Load model
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
data = pd.DataFrame([{'tenure': 12, 'MonthlyCharges': 70, ...}])
churn_prob = model.predict_proba(data)[0][1]
print(f"Churn Risk: {churn_prob:.1%}")
```

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip
- Git

### Steps

```bash
# 1. Clone
git clone https://github.com/ShashankRajput90/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Download dataset
# Visit: https://www.kaggle.com/blastchar/telco-customer-churn
# Place CSV in data/raw/

# Or use Kaggle API:
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/raw/
```

---

## 🚀 Roadmap

### Completed ✅

- [x] Full ML pipeline with 5 algorithms
- [x] Interactive Streamlit dashboard
- [x] Comprehensive documentation
- [x] Feature engineering & SMOTE

### In Progress 🔄

- [ ] FastAPI REST endpoint
- [ ] Unit test coverage >80%
- [ ] CI/CD with GitHub Actions

### Planned 📅

- [ ] Cloud deployment (AWS/Azure)
- [ ] SHAP values for explainability
- [ ] Customer segmentation module
- [ ] A/B testing framework

See [README_DETAILED.md](README_DETAILED.md) for full roadmap.

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork & clone
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git

# Create branch
git checkout -b feature/amazing-feature

# Make changes, test, commit
git commit -m "Add amazing feature"

# Push & create PR
git push origin feature/amazing-feature
```

---

## 👤 Author

**Shashank Lodhi**  
🎓 Data Science @ MIET'26  
📍 Meerut, Uttar Pradesh, India

[![GitHub](https://img.shields.io/badge/GitHub-ShashankRajput90-181717?logo=github)](https://github.com/ShashankRajput90)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-blue)](https://shashankrajput90.github.io/my-portfolio-repo/)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- Dataset: [IBM/Kaggle Telco Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Libraries: Scikit-learn, XGBoost, Streamlit communities
- MIET Faculty: Guidance and support

---

## 📊 Project Stats

![GitHub last commit](https://img.shields.io/github/last-commit/ShashankRajput90/customer-churn-prediction)
![GitHub repo size](https://img.shields.io/github/repo-size/ShashankRajput90/customer-churn-prediction)
![GitHub stars](https://img.shields.io/github/stars/ShashankRajput90/customer-churn-prediction?style=social)

---

<div align="center">

**Made with ❤️ for the Data Science Community**

[Report Bug](https://github.com/ShashankRajput90/customer-churn-prediction/issues) · [Request Feature](https://github.com/ShashankRajput90/customer-churn-prediction/issues) · [Discussions](https://github.com/ShashankRajput90/customer-churn-prediction/discussions)

*Last updated: November 2025*

</div>