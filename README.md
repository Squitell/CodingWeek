# Medical Decision Support Application

## Project Report

### Introduction

This project focuses on developing a decision-support application designed to assist physicians in predicting the success rate of pediatric bone marrow transplants. The primary goal is to create a reliable and explainable machine learning model that ensures transparency in medical predictions. By leveraging advanced AI techniques, we aim to enhance the decision-making process for medical professionals and ultimately improve patient outcomes.

### Objectives

The main objectives of this project are:

- Develop an explainable machine learning model to predict transplant success based on patient data.
- Ensure transparency through SHAP (SHapley Additive exPlanations)-based interpretability techniques, helping medical professionals understand the AI’s decision-making process.
- Build a user-friendly interface using Streamlit or Flask to provide an accessible and intuitive experience for users.
- Implement professional software development practices, including GitHub for version control and CI/CD automation for seamless deployment.
- Explore prompt engineering by documenting AI-generated prompts used in the workflow to refine model responses and improve accuracy.

### Dataset Used

The dataset used in this project is the **Bone Marrow Transplant Children Dataset**, which contains patient-specific features, medical history, and post-transplant outcomes. This data serves as a crucial factor for training and evaluating the machine learning model.

### Data Processing and Analysis

To ensure data quality and improve model performance, the following preprocessing steps were undertaken:

1. **Handling Missing Values**: Missing values were handled using mean imputation for numerical data and mode imputation for categorical data.
2. **Feature Selection**: A correlation analysis was conducted to identify the most significant features influencing transplant success.
3. **Data Normalization**: Continuous variables were normalized to ensure consistency across different scales.
4. **Train-Test Splitting**: The dataset was split into training and testing sets to evaluate model performance effectively.

### Model Development

The machine learning model was developed using Python and various AI libraries, including:

- **Scikit-learn** for baseline models such as logistic regression, decision trees, and random forests.
- **XGBoost** for improved predictive performance.
- **SHAP** for model interpretability, allowing clinicians to understand the importance of each feature in predicting transplant outcomes.
- **TensorFlow/Keras** for deep learning experiments to explore more complex patterns in the data.

### Model Evaluation

The following metrics were used to assess model performance:

- **Accuracy**: Measures the percentage of correctly classified instances.
- **Precision & Recall**: Evaluates the model's ability to correctly identify successful and unsuccessful transplants.
- **F1 Score**: Balances precision and recall for a more comprehensive evaluation.
- **ROC-AUC Score**: Assesses the model’s ability to distinguish between different outcome classes.

### Deployment & User Interface

A web-based application was developed to provide an interactive platform for clinicians. The interface was built using:

- **Streamlit** for a lightweight, interactive dashboard.
- **Flask** for backend API integration.
- **Docker** for containerized deployment to ensure scalability and portability.

### Conclusion

This project demonstrates the potential of AI-driven decision support systems in the medical field. By incorporating explainable machine learning techniques, we aim to build trust and transparency in AI-powered predictions, ultimately assisting physicians in making informed decisions for pediatric bone marrow transplant patients.

Future improvements include integrating real-time patient data updates, refining model accuracy with larger datasets, and expanding the application to other medical prediction scenarios.

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
