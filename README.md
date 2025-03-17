# Medical Decision Support Application

# Predicting Pediatric Bone Marrow Transplant Success

## Introduction

This project focuses on developing a decision-support application designed to assist physicians in predicting the success rate of pediatric bone marrow transplants. The primary goal is to create a reliable and explainable machine learning model that ensures transparency in medical predictions. By leveraging advanced AI techniques, we aim to improve the decision-making process for medical professionals and ultimately enhance patient outcomes.

## Objectives

The main objectives of this project are:

- Develop an explainable machine learning model to predict transplant success based on patient data.
- Ensure transparency using SHAP (SHapley Additive exPlanations) techniques to help medical professionals understand the AI’s decision-making process.
- Build an intuitive user interface using Streamlit or Flask to provide easy access for users.
- Implement best software development practices with version control and CI/CD automation.
- Explore prompt engineering to improve the AI-generated prompts for model performance refinement.

## Dataset Overview

The **Bone Marrow Transplant Children Dataset** was used, containing various patient-specific features, medical history, and post-transplant outcomes. This dataset is crucial for training and evaluating the machine learning model.

<<<<<<< Updated upstream
## Data Processing and Preprocessing
=======
📂 CodingWeek_Grp8/
│── 📂 .venv/                     # Virtual environment
│── 📂 Back_end/                  # Backend API and related files
│   │── main.py                  # FastAPI backend for handling predictions
│   │── requirements.txt          # Dependencies for backend
│── 📂 machine_learning/           # Machine learning pipeline
│   │── 📂 data/                   # Processed datasets used for training/testing
│   │── 📂 feature_importance_plots/  # Feature importance visualizations
│   │── 📂 imbalance_plots/         # Class imbalance handling visualizations
│   │── 📂 model_performance/       # Model evaluation results
│   │── 📂 models/                  # Trained ML models
│   │── 📂 notebooks/               # Exploratory data analysis (EDA) & experiments
│   │── 📂 plots/                   # Data visualization & SHAP analysis results
│   │── 📂 shap_analysis/           # SHAP interpretability analysis
│   │── 📂 src/                     # Machine learning scripts
│   │── 📂 tests/                   # Unit tests for ML components               
│── 📂 Front_end/                   # React frontend for user interaction
│── instructions.txt                # Steps to run the application
│── README.md                       # Project overview and theoretical documentation
>>>>>>> Stashed changes

Several preprocessing steps were implemented to enhance the model’s performance:

- **Handling Missing Values**: Imputation was used to address missing values—mean for numerical data and mode for categorical data.
- **Feature Selection**: A correlation analysis was conducted to identify the most significant features affecting transplant success.
- **Data Normalization**: Continuous variables were normalized for consistency across different scales.
- **Train-Test Splitting**: The dataset was split into training and testing sets to evaluate model performance effectively.

## Model Development

Machine learning models were developed using Python libraries such as:

- **Scikit-learn** for baseline models like logistic regression, decision trees, and random forests.
- **XGBoost** for enhanced predictive performance.
- **SHAP** for model interpretability, helping clinicians understand how each feature contributes to transplant success predictions.
- **TensorFlow/Keras** was explored for deep learning experiments to identify more complex patterns in the data.

## Model Evaluation

The following metrics were used to evaluate model performance:

- **Accuracy**: Proportion of correctly predicted instances.
- **Precision & Recall**: Measures the model’s ability to identify successful and unsuccessful transplants.
- **F1 Score**: A balanced metric between precision and recall.
- **ROC-AUC Score**: Assesses the model’s ability to distinguish between transplant outcome classes.

## Results

- **Class Imbalance**: The dataset was imbalanced. Techniques such as SMOTE or class weighting were applied to improve the balance, leading to improved model generalization and fairness in predictions.
- **Best Model Performance**: Random Forest was identified as the best-performing model, with the following metrics:
  - Train Accuracy: 1.0
  - Train ROC-AUC: 1.0
  - Validation Accuracy: 0.939
  - Validation ROC-AUC: 0.981 (highest among models)
- **SHAP Insights**: The top influential features according to SHAP were:
  - **Donor Age**: Younger donors led to better outcomes.
  - **Recipient CMV Status**: Negative CMV status improved transplant success.
  - **Stem Cell Source**: Peripheral blood stem cells resulted in better outcomes.

## Prompt Engineering Insights

Prompt engineering significantly contributed to the following improvements:

- **Improved SHAP Explanations**: The decision-making process of the model became clearer, helping clinicians better understand model predictions.
- **Refined Data Preprocessing & Feature Selection**: Better strategies for data preprocessing and feature selection were implemented, which enhanced model performance.
- **Enhanced Interpretability**: The insights helped in refining the model’s interpretability, fostering better communication between clinicians and the AI system.

## Deployment & User Interface

A user-friendly, web-based application was developed to provide an interactive platform for clinicians. The interface was built using:

- **Streamlit**: For creating an interactive dashboard.
- **Flask**: For backend API integration.
- **Docker**: For containerized deployment, ensuring scalability and portability.

## Conclusion

This project demonstrates the potential of AI-driven decision support systems in the medical field. By incorporating explainable machine learning techniques, we aim to foster transparency and build trust in AI-powered predictions, ultimately assisting physicians in making informed decisions for pediatric bone marrow transplant patients. 

**Future Improvements**:
- Integrating real-time patient data updates.
- Refining model accuracy with larger datasets.
- Expanding the application to other medical prediction scenarios.


---

## Prompt Engineering Documentation

### Task: Model Evaluation

#### Prompts Used

**Prompt 1**: Generate a function to evaluate a model on test data


Generate a function to evaluate a model on test data and return performance metrics.
## Result:

```python
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a model on the test set and return performance metrics.
    """
    X_test = reindex_features(X_test, model)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0)
    }
    return metrics
```

## Effectiveness of Prompts:

The prompt was effective in generating a function that evaluates a model on test data and returns performance metrics. The generated function includes the necessary steps to reindex features, make predictions, and calculate various performance metrics such as accuracy, ROC-AUC, precision, recall, and F1 score.

## Potential Improvements:

- **Prompt Specificity**: The prompt could be more specific by mentioning the exact metrics required or any additional functionality needed.
- **Error Handling**: The generated function could include more robust error handling to manage potential issues during evaluation.
- **Documentation**: Ensure that the generated function includes comprehensive docstrings and comments for better readability and maintainability.

## Improved Prompt:

Generate a function to evaluate a model on test data, including accuracy, ROC-AUC, precision, recall, and F1 score metrics. Ensure the function handles potential errors and includes detailed docstrings.


```python
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a model on the test set and return performance metrics.
    
    Parameters:
    - model: Trained model to be evaluated.
    - X_test: DataFrame containing test features.
    - y_test: Series containing test labels.
    
    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    try:
        X_test = reindex_features(X_test, model)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0)
        }
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        metrics = {}
    return metrics
```
In this project, prompt engineering played a crucial role in generating and refining the necessary functions and processes for model evaluation. By iteratively improving the clarity and specificity of the prompts, we were able to generate code that accurately loads test data, evaluates models, and computes essential performance metrics, such as accuracy, ROC-AUC, precision, recall, and F1 score. The effectiveness of the prompts was evident in the comprehensive and functional code produced, ensuring seamless integration of the models into the evaluation pipeline. Additionally, the prompts facilitated the incorporation of error handling and improved documentation, enhancing both the robustness and maintainability of the resulting code. Ultimately, prompt engineering allowed us to optimize the evaluation process and ensure that the models were assessed effectively against the test data, leading to a clearer understanding of their performance.

<<<<<<< Updated upstream
=======
-------📌 For setup and execution steps, refer to instructions.txt.-------
>>>>>>> Stashed changes
