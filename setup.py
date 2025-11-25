#!/usr/bin/env python3
"""
Setup script for Customer Churn Prediction package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="customer-churn-prediction",
    version="1.0.0",
    author="Shashank Lodhi",
    author_email="shashank.lodhi@example.com",
    description="ML-powered customer churn prediction system for telecom industry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShashankRajput90/customer-churn-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "xgboost>=1.7.0",
        "imbalanced-learn>=0.10.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.11.0",
        "streamlit>=1.15.0",
        "joblib>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.10.0",
            "flake8>=5.0.0",
            "pylint>=2.15.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "churn-train=src.churn_prediction_pipeline:main",
            "churn-dashboard=streamlit:run src/streamlit_app.py",
        ],
    },
)
