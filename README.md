# Medical Decision Support Application

## Project Report

### Introduction
This project focuses on developing a decision-support application to assist physicians in predicting the success rate of pediatric bone marrow transplants. The primary objective is to create a reliable and explainable machine learning model that ensures transparency in medical predictions. By leveraging advanced AI techniques, we aim to enhance the decision-making process for medical professionals, ultimately improving patient outcomes.

### Objectives
The main objectives of this project are:

- **Developing** an explainable machine learning model to predict transplant success based on patient data.
- **Ensuring transparency** through SHAP (SHapley Additive exPlanations)-based interpretability techniques, which help medical professionals understand the AI’s decision-making process.
- **Building** a user-friendly interface using Streamlit or Flask to provide an accessible and intuitive experience for users.
- **Implementing** professional software development practices, including GitHub for version control and CI/CD automation for seamless deployment.
- **Exploring prompt engineering** by documenting AI-generated prompts used in the workflow to refine model responses and improve accuracy.

### Dataset Used
The dataset utilized in this project is the **Bone Marrow Transplant Children Dataset**. This dataset contains patient-specific features, medical history, and post-transplant outcomes, which serve as crucial factors in training and evaluating the machine learning model.

### Data Processing and Analysis
To ensure data quality and improve model performance, the following preprocessing steps were undertaken:
- **Handling Missing Values**: The dataset was evaluated for missing values, and appropriate techniques were used to handle them, such as mean imputation for numerical data and mode imputation for categorical data.
- **Feature Selection**: A correlation analysis was conducted to identify the most significant features influencing transplant success.
- **Data Normalization**: Continuous variables were normalized to ensure consistency across different scales.
- **Train-Test Splitting**: The dataset was split into training and testing sets to evaluate model performance effectively.

### Model Development
The machine learning model was developed using Python and various AI libraries, including:
- **Scikit-learn** for baseline models such as logistic regression, decision trees, and random forests.
- **XGBoost** for improved predictive performance.
- **SHAP** for model interpretability, allowing clinicians to understand the importance of each feature in predicting transplant outcomes.
- **TensorFlow/Keras** for deep learning experiments to explore more complex patterns in the data.

### Model Evaluation
To assess model performance, the following metrics were used:
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

### References
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpretable Model Predictions.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
- TensorFlow Team (2023). TensorFlow Documentation.

