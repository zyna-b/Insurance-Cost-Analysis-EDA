# 📊 Insurance Cost Prediction - Comprehensive EDA & Statistical Analysis

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-orange.svg)](https://seaborn.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-red.svg)](https://scikit-learn.org/)

## 🎯 Project Overview

This repository contains a **comprehensive Exploratory Data Analysis (EDA)** of insurance cost data to identify key factors affecting medical insurance charges. The analysis includes data preprocessing, statistical testing, feature engineering, and correlation analysis to understand relationships between demographics, health factors, and insurance costs.

## 📈 Key Findings & Insights

- **Smoking Status**: Strongest predictor of insurance charges (highest correlation)
- **BMI Categories**: Obesity significantly impacts insurance costs
- **Regional Differences**: Southeast region shows notable cost variations
- **Age Factor**: Positive correlation with insurance charges
- **Statistical Significance**: Chi-square tests confirm categorical variable relationships

## 🔍 Analysis Highlights

### Data Exploration
- **Dataset Size**: 1,338 insurance records
- **Features**: Age, BMI, Children, Sex, Smoker Status, Region
- **Target Variable**: Insurance Charges
- **Data Quality**: No missing values, 1 duplicate removed

### Statistical Methods Applied
- **Pearson Correlation Analysis**: Quantifying linear relationships
- **Chi-Square Tests**: Testing independence of categorical variables
- **Feature Engineering**: BMI categorization, dummy encoding
- **Data Standardization**: StandardScaler for numerical features

### Visualizations Created
- Distribution plots for numerical variables
- Count plots for categorical features
- Correlation heatmaps
- Box plots for outlier detection
- Feature relationship analysis

## 🛠️ Technologies & Libraries

```python
# Core Data Science Stack
import pandas as pd           # Data manipulation
import numpy as np            # Numerical computing
import seaborn as sns         # Statistical visualization
import matplotlib.pyplot as plt  # Plotting

# Statistical Analysis
from scipy.stats import pearsonr, chi2_contingency

# Machine Learning
from sklearn.preprocessing import StandardScaler
```

## 📁 Repository Structure

```
insurance-eda/
├── insurance.ipynb          # Main analysis notebook
├── insurance.csv           # Dataset
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/insurance-eda.git
cd insurance-eda
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook insurance.ipynb
```

### Dependencies
```txt
pandas>=1.3.0
numpy>=1.21.0
seaborn>=0.11.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

## 📊 Data Analysis Workflow

### 1. Data Exploration & Cleaning
- Initial data inspection and shape analysis
- Missing value detection and handling
- Duplicate removal and data type verification

### 2. Exploratory Data Analysis
- **Univariate Analysis**: Distribution of individual variables
- **Bivariate Analysis**: Relationships between features and target
- **Correlation Analysis**: Identifying linear relationships

### 3. Feature Engineering
- BMI categorization (Underweight, Normal, Overweight, Obesity)
- Categorical encoding (Gender, Smoker status, Region)
- Feature standardization for analysis

### 4. Statistical Testing
- **Correlation Analysis**: Pearson correlation coefficients
- **Chi-Square Tests**: Independence testing for categorical variables
- **Feature Selection**: Identifying significant predictors

## 🎯 Business Insights

### Primary Cost Drivers
1. **Smoking Status**: Most significant factor affecting insurance costs
2. **BMI Category**: Obesity increases insurance charges substantially
3. **Regional Factors**: Geographic location impacts pricing
4. **Age Demographics**: Older individuals face higher charges

### Recommendations
- **Risk Assessment**: Focus on smoking and obesity factors
- **Regional Pricing**: Consider location-based premium adjustments
- **Health Programs**: Implement wellness initiatives for high-risk groups

## 📈 Feature Importance Results

| Feature | Correlation with Charges | Statistical Significance |
|---------|-------------------------|-------------------------|
| Smoking Status | Highest | ✅ Significant |
| BMI Category | High | ✅ Significant |
| Age | Moderate | ✅ Significant |
| Region | Moderate | ✅ Significant |
| Gender | Low | ✅ Significant |

## 🔬 Statistical Methods Explained

### Correlation Analysis
- **Pearson Correlation**: Measures linear relationship strength (-1 to +1)
- **Interpretation**: Values closer to ±1 indicate stronger relationships

### Chi-Square Testing
- **Purpose**: Tests independence between categorical variables
- **Application**: Validates relationships between demographics and cost categories
- **Significance Level**: α = 0.05

## 📝 Future Enhancements

- [ ] Predictive modeling implementation
- [ ] Advanced feature engineering techniques
- [ ] Interactive visualizations with Plotly
- [ ] Machine learning model comparison
- [ ] Cross-validation and model evaluation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@zyna-b](https://github.com/zyna-b)
- LinkedIn: [Zainab Hamid](https://linkedin.com/in/zainab-hamid-187a18321/)

## 🙏 Acknowledgments

- Dataset source: [Insurance Cost Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Inspiration from data science community best practices
- Statistical methods from scipy documentation

---

⭐ **Star this repository if you found it helpful!**
