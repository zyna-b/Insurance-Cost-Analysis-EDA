# 🏥💰 Insurance Cost Analysis & Prediction - Complete Data Science Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A comprehensive exploratory data analysis (EDA) and machine learning project for predicting insurance costs using demographic and health factors.**

## 🎯 Project Overview

This project demonstrates a complete data science workflow for analyzing and predicting insurance costs. Using a dataset of 1,338 insurance records, we explore relationships between demographic factors, health indicators, and insurance charges through advanced statistical analysis and machine learning.

### 🔍 Key Features
- **Complete EDA Pipeline**: From data exploration to feature engineering
- **Statistical Analysis**: Correlation analysis, Chi-square testing, and hypothesis testing
- **Machine Learning Model**: Linear regression with 80.4% R-squared accuracy
- **Feature Engineering**: BMI categorization and advanced feature selection
- **Data Visualization**: Professional plots using Matplotlib and Seaborn

## 📊 Dataset Information

| Feature | Description | Type |
|---------|-------------|------|
| **Age** | Age of primary beneficiary | Numerical (18-64) |
| **Sex** | Insurance contractor gender | Categorical (male/female) |
| **BMI** | Body mass index | Numerical (15.96-53.13) |
| **Children** | Number of dependents | Numerical (0-5) |
| **Smoker** | Smoking status | Categorical (yes/no) |
| **Region** | Beneficiary's residential area | Categorical (4 regions) |
| **Charges** | Medical costs billed by insurance | Target Variable |

**Dataset Stats**: 1,338 records • 7 features • No missing values • 1 duplicate removed

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
Jupyter Notebook or JupyterLab
```

### Installation & Setup

1. **Clone this repository**
```bash
git clone https://github.com/zyna-b/Insurance-Cost-Analysis-EDA.git
cd Insurance-Cost-Analysis-EDA
```

2. **Create virtual environment**
```bash
python -m venv venv_py39
# Windows
venv_py39\Scripts\activate
# macOS/Linux
source venv_py39/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch analysis**
```bash
jupyter notebook insurance.ipynb
```

## 📈 Analysis Workflow

### 1. 🔍 Exploratory Data Analysis
- **Data Inspection**: Shape, types, missing values, duplicates
- **Descriptive Statistics**: Central tendencies and distributions
- **Univariate Analysis**: Individual feature distributions
- **Bivariate Analysis**: Feature relationships with target variable

### 2. 📊 Data Visualization
- **Distribution Plots**: Age, BMI, children, charges histograms with KDE
- **Count Plots**: Categorical variable frequencies
- **Box Plots**: Outlier detection and quartile analysis
- **Correlation Heatmap**: Feature relationship visualization

### 3. 🧹 Data Preprocessing
- **Data Cleaning**: Duplicate removal and type conversion
- **Feature Encoding**: 
  - Binary encoding for gender and smoker status
  - One-hot encoding for region variables
- **Feature Engineering**: BMI categorization (Underweight, Normal, Overweight, Obesity)
- **Standardization**: StandardScaler for numerical features

### 4. 📊 Statistical Analysis

#### Correlation Analysis
```python
# Key findings from Pearson correlation analysis
correlations = {
    'is_smoker': 0.787,      # Strongest predictor
    'age': 0.299,            # Moderate positive correlation
    'bmi': 0.198,            # Weak positive correlation
    'children': 0.068,       # Very weak correlation
    # ... additional features
}
```

#### Chi-Square Testing
- **Purpose**: Test independence between categorical variables and charges
- **Significance Level**: α = 0.05
- **Results**: Identified significant features for model inclusion

### 5. 🤖 Machine Learning Model

#### Linear Regression Implementation
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Model training and evaluation
model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
```

#### Model Performance
- **R-squared Score**: 0.804 (80.4% variance explained)
- **Adjusted R-squared**: 0.799
- **Train-Test Split**: 80% training, 20% testing
- **Random State**: 42 (reproducible results)

## 🔍 Key Insights & Findings

### 💡 Business Intelligence
1. **Smoking Impact**: Smoking status is the strongest predictor of insurance costs
2. **Age Factor**: Older individuals tend to have higher insurance charges
3. **BMI Influence**: Higher BMI correlates with increased medical costs
4. **Regional Variations**: Geographic location affects insurance pricing
5. **Family Size**: Number of children has minimal impact on costs

### 📊 Statistical Discoveries
- **Correlation Strength**: Smoking status shows 0.787 correlation with charges
- **Feature Importance**: Age, BMI, and smoking status are primary cost drivers
- **Data Distribution**: Charges show right-skewed distribution (typical for insurance data)
- **Gender Impact**: Minimal difference in average costs between males and females

## 🛠️ Technologies & Libraries

### Core Stack
```python
import pandas as pd              # Data manipulation and analysis
import numpy as np               # Numerical computing
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns           # Statistical data visualization
```

### Machine Learning & Statistics
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, chi2_contingency
```

## 📁 Project Structure

```
Insurance-Cost-Analysis-EDA/
├── 📓 insurance.ipynb          # Main analysis notebook
├── 📊 insurance.csv           # Dataset (1,338 records)
├── 📋 requirements.txt        # Python dependencies
├── 📄 README.md              # Project documentation
└── 📁 venv_py39/             # Virtual environment
    ├── Scripts/              # Environment executables
    ├── Lib/                  # Installed packages
    └── Include/              # Header files
```

## 📈 Visualizations Included

- **Distribution Analysis**: Histograms with KDE for numerical variables
- **Categorical Analysis**: Count plots for sex, smoker, region, children
- **Correlation Matrix**: Heatmap showing feature relationships
- **Box Plots**: Outlier detection for numerical features
- **Feature Engineering**: BMI categorization visualization

## 🔬 Statistical Methods Explained

### Correlation Analysis
- **Pearson Correlation**: Measures linear relationship strength (-1 to +1)
- **Interpretation**: Values closer to ±1 indicate stronger linear relationships
- **Application**: Identifying features most correlated with insurance charges

### Chi-Square Testing
- **Purpose**: Tests independence between categorical variables and target
- **Null Hypothesis**: Variables are independent
- **Decision Rule**: Reject H0 if p-value < 0.05
- **Business Value**: Validates which categorical features significantly impact costs

### Feature Engineering
- **BMI Categories**: Medical standard classifications
- **Dummy Variables**: Binary encoding for categorical features
- **Standardization**: Zero mean, unit variance for numerical features

## 🎯 Model Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R-squared** | 0.804 | Model explains 80.4% of variance |
| **Adjusted R-squared** | 0.799 | Accounts for number of predictors |
| **Features Used** | 7 | Optimal feature subset selected |
| **Sample Size** | 1,337 | After duplicate removal |

## 🔮 Future Enhancements

- [ ] **Advanced Models**: Random Forest, Gradient Boosting, Neural Networks
- [ ] **Cross-Validation**: K-fold validation for robust performance metrics
- [ ] **Feature Engineering**: Polynomial features, interaction terms
- [ ] **Hyperparameter Tuning**: Grid search for optimal parameters
- [ ] **Interactive Dashboard**: Streamlit or Dash implementation
- [ ] **Model Deployment**: Flask API for real-time predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset Source**: [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Statistical Methods**: Scipy documentation and best practices
- **Visualization Inspiration**: Seaborn gallery and matplotlib examples
- **Machine Learning Techniques**: Scikit-learn documentation

## 👨‍💻 Author

**Zainab Hamid**
- 🐙 GitHub: [@zyna-b](https://github.com/zyna-b)
- 💼 LinkedIn: [Zainab Hamid](https://linkedin.com/in/zainab-hamid-187a18321/)
- 📧 Email: [Contact for collaborations]

## 📊 Keywords

`insurance-analysis` `data-science` `machine-learning` `exploratory-data-analysis` `python` `pandas` `scikit-learn` `statistical-analysis` `data-visualization` `linear-regression` `feature-engineering` `correlation-analysis` `chi-square-testing` `jupyter-notebook` `healthcare-analytics`

---

⭐ **Found this project helpful? Please consider starring the repository!**

🔍 **Looking for specific analysis techniques? Check out the detailed Jupyter notebook for complete implementation.**

📈 **Interested in similar projects? Follow for more data science content!**