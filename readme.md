# Predictive Modeling for Diabetes using ML Classifiers 🩺

## Project Overview

This project aims to develop and evaluate various machine learning models to predict diabetes based on a set of medical and demographic features. By leveraging different classifiers and optimizing their hyperparameters, we aim to identify the most effective model for accurate and reliable diabetes prediction.

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Data Collection](#data-collection)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Feature Engineering](#feature-engineering)
6. [Model Selection & Training](#model-selection-training)
7. [Model Evaluation](#model-evaluation)
8. [Conclusion](#conclusion)
9. [Installation](#installation)
10. [Usage](#usage)
11. [License](#license)

## Problem Definition

### Project Goal

The goal of this project is to develop and evaluate various machine learning models to predict diabetes based on a set of medical and demographic features.

### Problem Statement

Diabetes is a chronic disease that affects millions of people worldwide. Early detection and management are crucial to prevent severe complications. The problem at hand is to build a predictive model that can accurately classify individuals as diabetic or non-diabetic based on their medical and demographic data.

### Impact of the Solution

An accurate predictive model for diabetes can significantly impact public health by enabling early diagnosis and intervention. This can lead to better management of the disease, reduced healthcare costs, and improved quality of life for individuals at risk.

## Data Collection

### Source of Data

The dataset is sourced from Priyam Choksi on Kaggle - Comprehensive Diabetes Clinical Dataset (100k rows), which can be found at the following URL: [Kaggle Dataset](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset)

### Composition of the Dataset

The dataset contains health-related information about individuals, including year, gender, age, location, race, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level, and diabetes status.

## Data Preprocessing

1. **Loading the Dataset and Handling Missing Values**
2. **Encoding Categorical Variables**
3. **Standardizing Numerical Variables**
4. **Handling Class Imbalance using SMOTE**

## Exploratory Data Analysis (EDA)

1. **Descriptive Statistics: Summary Statistics for Numerical Features**
2. **Data Visualization**
   - Heatmap of features correlation
   - Class Distribution
   - Distribution of Numerical Features

## Feature Engineering

1. **Removing Features with Low Correlation with Diabetes**

## Model Selection & Training

1. **Logistic Regression Classifier**
2. **K-Nearest Neighbors Classifier**
3. **RandomForest Classifier**
4. **XGBoost Classifier**
5. **CatBoost Classifier**
6. **Gaussian Naive Bayes Classifier**
7. **Modular Neural Network**

## Model Evaluation

1. **Accuracy**
2. **F1 Score**
3. **Precision & Recall**
4. **Radar Chart for Overall Performance**

## Conclusion

In this project, we developed and evaluated several machine learning models to predict diabetes using a dataset with significant class imbalance. Through comprehensive data preprocessing including handling missing values, encoding categorical variables, standardizing numerical features, and addressing class imbalance with SMOTE, we ensured the dataset was well-prepared for modeling. We explored a variety of classifiers, including Logistic Regression, K-Nearest Neighbors, RandomForest, XGBoost, CatBoost, Gaussian Naive Bayes, and a custom Neural Network. Our evaluation metrics—accuracy, precision, recall, and F1-score—revealed that XGBoost and CatBoost consistently outperformed other models, demonstrating high accuracy and balanced performance across both classes. RandomForest also showed strong results, while Logistic Regression and Gaussian Naive Bayes struggled with the class imbalance. The neural network, optimized using Optuna, performed well but did not surpass the tree-based models. Overall, the project highlights the importance of addressing class imbalance. Further considerations include selecting a different strategy for handling class imbalance, for instance, hybrid sampling (e.g. SMOTEENN), class weighting or ensemble methods.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset) and place it in the project directory.

2. Run the Jupyter notebook or Python script to preprocess the data, train the models, and evaluate their performance.

3. To resume the Optuna study, use the `resume_study` function provided in the script.

## License

This project is licensed under the Apache v2.0 License. See the [LICENSE](LICENSE) file for more details.

---
The link to the drive is available upon requst.
Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback and contributions are highly appreciated!
