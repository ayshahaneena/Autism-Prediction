# Autism-Prediction
![Autism Model](https://www.osfhealthcare.org/blog/wp-content/uploads/2019/05/choi-autism-ft5.jpg)


# Autism Prediction using Machine Learning

This project aims to predict whether a patient is diagnosed with Autism Spectrum Disorder (ASD) based on a variety of features such as scores from an Autism Spectrum Quotient (AQ) screening tool, demographic information, and family history.

## Problem Statement
The goal of this project is to create a model that classifies whether a patient has Autism (Class/ASD = 1) or not (Class/ASD = 0) based on the available features.

## Features
The dataset consists of the following columns:

- **ID**: ID of the patient
- **A1_Score to A10_Score**: Scores based on the Autism Spectrum Quotient (AQ) 10-item screening tool
- **age**: Age of the patient in years
- **gender**: Gender of the patient
- **ethnicity**: Ethnicity of the patient
- **jaundice**: Whether the patient had jaundice at the time of birth
- **autism**: Whether an immediate family member has been diagnosed with autism
- **country_of_res**: Country of residence of the patient
- **used_app_before**: Whether the patient has undergone a screening test before
- **result**: Score for AQ1-10 screening test
- **age_desc**: Age description of the patient
- **relation**: Relation of the person who completed the test
- **Class/ASD**: The target column, where 0 represents "No" and 1 represents "Yes" (indicating autism diagnosis)

## Steps Taken

### 1. Data Cleaning
- Checked for missing values and outliers.
- Handled missing data and dropped irrelevant columns.

### 2. Exploratory Data Analysis (EDA)
Key insights obtained during the EDA:
- The mean age of the patients is 28.
- White European patients are significantly more likely to have autism.
- Patients who had jaundice at birth show a higher tendency to be diagnosed with autism.
- Patients with a family history of autism are more likely to have autism.
- The dataset has 76% non-autistic and 23% autistic individuals.
- There is a balanced gender distribution with 51% male and 48% female patients.

### 3. Data Preprocessing
- Encoded categorical variables.
- Visualized the correlation between features using a heatmap.
- Balanced the dataset using upsampling to handle class imbalance.
- Scaled the features to improve model performance.

### 4. Model Selection and Evaluation
I used multiple classifiers to predict autism:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier
- AdaBoost Classifier
- SGD Classifier
- XGBoost Classifier
- Naive Bayes Classifier

After comparing the models using accuracy and ROC-AUC scores, **Random Forest** performed the best in terms of classification accuracy and ROC-AUC score.

### 5. Feature Importance
- Identified the most important features affecting autism diagnosis using the Random Forest model.

### 6. Model Saving
- Saved the best-performing Random Forest model using **Pickle** for later use and deployment.

### 7. **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
- Precision, Recall, and F1-Score
- **ROC-AUC Score**: Measures the model's ability to distinguish between classes, with a higher value indicating better performance.

## Conclusion
- The **Random Forest Classifier** outperformed all other models in terms of accuracy and ROC-AUC score.
- Key features like Result, age, country of residence, family history of autism, jaundice at birth, and ethnicity play an important role in predicting autism.

## Usage
1. Clone the repository.
2. Install required libraries from `requirements.txt`.
3. Load the saved Random Forest model using Pickle for predictions.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Dataset Source: [[kaggle](https://www.kaggle.com/competitions/autism-prediction/data)]
## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `pickle`



