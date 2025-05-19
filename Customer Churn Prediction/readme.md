# Customer Churn Prediction via Customer Segmentation

## Project Overview

This project focuses on predicting customer churn for a telecommunications company. It employs a two-stage approach:

1. **Customer Segmentation**: Utilizes K-Means clustering to group customers based on their `tenure`, `MonthlyCharges`, and `TotalCharges`. This segmentation aims to identify distinct customer behaviors and characteristics.
2. **Churn Prediction**: Develops and evaluates machine learning models (Logistic Regression and Gradient Boosting) to predict the likelihood of a customer churning. The cluster assignments from the segmentation step are used as an additional feature for the prediction models. SMOTE (Synthetic Minority Over-sampling TEchnique) is applied to address class imbalance in the churn data.

The primary goal is to provide actionable insights into customer churn, enabling the company to implement targeted retention strategies.

## Dataset

The project utilizes the **"WA_Fn-UseC_-Telco-Customer-Churn.csv"** dataset. This dataset contains various attributes for each customer, including:

- Demographic information
- Services subscribed to (e.g., PhoneService, InternetService, OnlineSecurity)
- Account information (e.g., tenure, contract, payment method, monthly charges, total charges)
- Churn status (whether the customer has churned or not)

## Dependencies

The following Python libraries are required to run this project:

- `pandas` (for data manipulation and analysis)
- `numpy` (for numerical operations)
- `scikit-learn` (for machine learning tasks including clustering, classification, preprocessing, and metrics)
- `imblearn` (for handling imbalanced datasets, specifically SMOTE)
- `matplotlib` (for generating visualizations)

Install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn imblearn matplotlib
```

## Methodology

### 1. Data Preprocessing
- **Loading Data**: The dataset is loaded into a pandas DataFrame.
- **Handling Missing/Incorrect Values**: The TotalCharges column is converted to a numeric type, and any resulting NaN values are imputed with 0.
- **Feature Engineering**:
  - Drop the customerID column.
  - Apply LabelEncoder to transform categorical features into numerical values.

### 2. Customer Segmentation (K-Means Clustering)
- **Feature Selection for Clustering**: tenure, MonthlyCharges, and TotalCharges
- **Scaling**: Standardize features using StandardScaler.
- **Determining Optimal Number of Clusters (k)**:
  - Use the Elbow Method with k from 1 to 10.
  - Choose k=5 based on the elbow point.
- **K-Means Model Training**: Train a K-Means model with k=5.
- **Cluster Assignment**: Assign each customer a cluster label and add it as a new feature.
- **Cluster Analysis**: Analyze mean values of clustering features per segment.
- **Visualization**: Generate a 3D scatter plot of customer clusters.

### 3. Churn Prediction Modeling
- **Feature Set**: All preprocessed features, including the cluster label.
- **Scaling**: Scale the feature set using StandardScaler.
- **Train-Test Split**: 80% training, 20% testing.
- **Handling Class Imbalance**: Apply SMOTE to training data only.
- **Model Training**:
  - Logistic Regression
  - Gradient Boosting Classifier
- **Model Evaluation**:
  - Classification Report: Precision, recall, F1-score, and support.
  - AUC-ROC Score
  - Feature Importance (for Gradient Boosting): Visualize with a bar chart.

## How to Run the Model

### Clone the Repository (if applicable):
```bash
git clone https://github.com/wahid18-maqs/ML_Models.git
cd ML_Models/Customer Churn Prediction
```

### Install Dependencies:
```bash
pip install pandas numpy scikit-learn imblearn matplotlib
```

### Prepare Dataset:
Ensure the WA_Fn-UseC_-Telco-Customer-Churn.csv file is in the same directory as the Jupyter Notebook (churn_model.ipynb) or update the file path in the notebook.

### Run the Notebook:
- Open and execute churn_model.ipynb cell-by-cell in a Jupyter environment.

The notebook will:
- Preprocess data
- Perform segmentation and save:
  - elbow_curve.png
  - cluster_3d_plot.png
- Train and evaluate churn prediction models
- Save classification results and:
  - feature_importance_fixed.png
- Provide an interactive prompt to enter a customerID and get a churn prediction using the Gradient Boosting model.

## Results Summary

### Logistic Regression Performance (with SMOTE):
```
              precision    recall  f1-score   support
           0       0.92      0.74      0.82      1036
           1       0.53      0.83      0.65       373
    accuracy                           0.76      1409
   macro avg       0.73      0.78      0.73      1409
weighted avg       0.82      0.76      0.77      1409

AUC-ROC: 0.8610
```

### Gradient Boosting Classifier Performance (with SMOTE):
```
              precision    recall  f1-score   support
           0       0.90      0.81      0.85      1036
           1       0.59      0.75      0.66       373
    accuracy                           0.79      1409
   macro avg       0.74      0.78      0.76      1409
weighted avg       0.82      0.79      0.80      1409

AUC-ROC: 0.8574
```

Both models perform reasonably well in identifying customers likely to churn, with SMOTE significantly improving recall. Gradient Boosting slightly outperforms Logistic Regression in overall accuracy and F1-score.

## Visualizations Generated
- **elbow_curve.png**: Elbow method to select optimal k.
- **cluster_3d_plot.png**: 3D scatter plot of customer clusters.
- **feature_importance_fixed.png**: Bar chart of feature importances (Gradient Boosting).

## Future Work
- Experiment with other clustering algorithms.
- Explore advanced feature engineering techniques.
- Hyperparameter tuning for models.
- Deploy as a web service or integrate into a CRM system.
