# 🏠 House Price Prediction

A machine learning project that predicts residential house prices using property-related features such as lot area, house condition, construction year, basement area, zoning, building type, and exterior material.

The project follows a complete machine learning workflow including **Exploratory Data Analysis (EDA), data preprocessing, feature engineering, model training, model comparison, cross-validation, and prediction**.

---

## 📌 Project Overview

Accurately estimating the price of a house is a regression problem where multiple characteristics of a property influence its market value.

In this project, machine learning models are trained to learn the relationship between property characteristics and **SalePrice**.

The primary objective is to build a robust regression model that can predict house prices for previously unseen properties.

### Problem Type

**Supervised Machine Learning → Regression**

### Target Variable

`SalePrice`

---

## 🎯 Objectives

* Perform detailed exploratory data analysis.
* Identify important patterns and relationships in the dataset.
* Handle missing values and categorical variables.
* Engineer meaningful features from existing variables.
* Apply logarithmic transformation to the target variable.
* Train multiple regression algorithms.
* Compare model performance using appropriate evaluation metrics.
* Perform cross-validation.
* Tune the best-performing model.
* Generate predictions for unseen test data.

---

## 📊 Dataset

The dataset contains information about residential properties.

### Dataset Files

```text
├── train.csv
├── test.csv
└── HousePricePrediction(1).csv
```

The training dataset contains the target variable `SalePrice`, while the test dataset is used for generating final predictions.

### Features

| Feature        | Description                       |
| -------------- | --------------------------------- |
| `Id`           | Unique property identifier        |
| `MSSubClass`   | Type/class of dwelling            |
| `MSZoning`     | General zoning classification     |
| `LotArea`      | Lot size in square feet           |
| `LotConfig`    | Configuration of the property lot |
| `BldgType`     | Type of dwelling                  |
| `OverallCond`  | Overall condition of the house    |
| `YearBuilt`    | Original construction year        |
| `YearRemodAdd` | Year of remodeling                |
| `Exterior1st`  | Primary exterior material         |
| `BsmtFinSF2`   | Finished basement area            |
| `TotalBsmtSF`  | Total basement area               |
| `SalePrice`    | Target house sale price           |

> **Note:** This project uses a reduced subset of the original Ames Housing dataset. Therefore, model performance is limited compared with models trained using the complete feature set.

---

## 🔍 Exploratory Data Analysis

The project performs several EDA steps to understand the dataset.

### Data Inspection

* Dataset shape
* Data types
* Statistical summary
* Missing values
* Duplicate records
* Unique categorical values

### Univariate Analysis

Distribution plots are used to understand:

* Numerical feature distributions
* Categorical feature frequencies
* Distribution of `SalePrice`

### Bivariate Analysis

Relationships between individual features and house prices are analyzed using:

* Scatter plots
* Box plots
* Distribution plots

### Correlation Analysis

A correlation matrix is used to identify relationships between numerical variables and `SalePrice`.

---

## 🛠️ Feature Engineering

Several additional features are created to improve the predictive capability of the models.

### Remodeling Age

```text
RemodelAge = YearRemodAdd - YearBuilt
```

This represents the difference between the construction year and remodeling year.

### Total Finished Basement

```text
TotalFinishedBasement = TotalBsmtSF + BsmtFinSF2
```

### Log Lot Area

```text
LogLotArea = log(1 + LotArea)
```

### Log Basement Area

```text
LogTotalBsmtSF = log(1 + TotalBsmtSF)
```

### Interaction Features

Additional interaction features are created between variables such as:

```text
OverallCond × YearBuilt
LotArea × OverallCond
```

These features allow the model to capture relationships that may not be obvious from individual variables.

---

## 🎯 Target Transformation

House prices are typically right-skewed.

Therefore, instead of directly modeling:

```text
SalePrice
```

the project uses:

```python
np.log1p(SalePrice)
```

as the training target.

Predictions are converted back to the original price scale using:

```python
np.expm1(prediction)
```

This transformation generally makes the target distribution more suitable for regression models and aligns well with the evaluation approach used in the Kaggle House Prices competition.

---

## 🤖 Machine Learning Models

Several algorithms are evaluated.

### 1. CatBoost Regressor

CatBoost is used as the primary model because it handles categorical variables effectively and performs well on small and medium-sized tabular datasets.

### 2. LightGBM

LightGBM is evaluated as a gradient boosting alternative with efficient tree-based learning.

### 3. XGBoost

XGBoost is another powerful gradient boosting algorithm used for comparison.

### 4. Gradient Boosting Regressor

A traditional Gradient Boosting model is used as an additional baseline.

---

## 📈 Model Evaluation

The models are evaluated using:

### RMSE

Root Mean Squared Error:

```text
RMSE = √(mean((y - ŷ)²))
```

Lower RMSE indicates better performance.

Because the target is log-transformed, the primary validation metric is calculated on:

```text
log(1 + SalePrice)
```

### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual values.

### R² Score

R² measures how much of the variance in the target variable is explained by the model.

Higher R² is better.

---

## 🔄 Cross-Validation

A **5-fold cross-validation** strategy is used to obtain a more reliable estimate of model performance.

The dataset is divided into five folds.

Each fold is used once as the validation set while the remaining four folds are used for training.

The final cross-validation score is calculated as the mean performance across all folds.

---

## ⚙️ Hyperparameter Tuning

After comparing multiple algorithms, the best-performing model can be further optimized.

For CatBoost, important hyperparameters include:

```text
iterations
learning_rate
depth
l2_leaf_reg
```

Hyperparameter tuning helps find a better balance between:

* Underfitting
* Overfitting
* Training time
* Generalization performance

---

## 🏆 Final Model

The final model is selected based on validation and cross-validation performance.

The selected model is then retrained using the complete training dataset before generating predictions for the test dataset.

The final predictions are converted from the logarithmic scale back to the original house-price scale.

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── models/
│   └── final_model.pkl
│
├── outputs/
│   └── submission.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💻 Technologies Used

### Programming Language

* Python

### Data Manipulation

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* CatBoost
* XGBoost
* LightGBM

### Development Environment

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/house-price-prediction.git
```

Navigate to the project directory:

```bash
cd house-price-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/house_price_prediction.ipynb
```

Run the notebook cells sequentially.

---

## 📋 Requirements

A `requirements.txt` file can contain:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
jupyter
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📊 Prediction Workflow

```text
Raw Dataset
     │
     ▼
Data Inspection
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Categorical Encoding
     │
     ▼
Target Log Transformation
     │
     ▼
Train / Validation Split
     │
     ▼
Model Training
     │
     ├── CatBoost
     ├── LightGBM
     ├── XGBoost
     └── Gradient Boosting
     │
     ▼
Model Comparison
     │
     ▼
Cross Validation
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Final Model
     │
     ▼
Test Prediction
     │
     ▼
submission.csv
```

---

## 📌 Key Learnings

This project demonstrates practical understanding of:

* Regression problems
* Exploratory Data Analysis
* Data preprocessing
* Handling categorical variables
* Feature engineering
* Target transformation
* Gradient boosting algorithms
* Model evaluation
* Cross-validation
* Hyperparameter tuning
* Model selection
* Kaggle-style prediction pipelines

---

## 🚀 Future Improvements

The current dataset contains only a subset of the features available in the complete Ames Housing dataset.

Future improvements include:

* Using the complete Ames Housing dataset.
* Adding important features such as `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `Neighborhood`, and `KitchenQual`.
* Performing more extensive hyperparameter optimization.
* Using ensemble/stacking techniques.
* Applying advanced outlier detection.
* Experimenting with feature selection.
* Building an interactive prediction application using Streamlit.
* Deploying the trained model as a web application or API.

---

## 📜 License

This project is intended for educational and learning purposes.

The dataset should be used according to the license and usage terms associated with its original source.

---

## 👩‍💻 Author

**Anayata Sinha**

B.Tech — Artificial Intelligence & Machine Learning

GitHub: `https://github.com/anayatasinha`

---

