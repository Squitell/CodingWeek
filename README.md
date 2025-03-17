Predicting Bone Marrow Transplant (BMT) Outcomes

➤ 1. Overview

This project aims to predict patient survival outcomes after Bone Marrow Transplantation (BMT) using machine learning models. By leveraging clinical and laboratory data, our goal is to provide a data-driven approach to assist medical professionals in assessing post-transplant prognosis.

➤ 2. Objectives

Develop a machine learning pipeline for predicting survival outcomes.

Address class imbalance in medical datasets.

Utilize explainability techniques (SHAP) to interpret model predictions.

Provide a front-end application for user interaction with predictions.

➤ 3. Project Structure

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


➤ 4. Machine Learning Components

🔹 4.1 Data Preprocessing

Handling missing values.

One-hot encoding categorical variables.

Splitting data into training and testing sets.

🔹 4.2 Class Imbalance Handling

Oversampling techniques like SMOTE.

Comparison of different resampling methods.

🔹 4.3 Model Training

Models trained: RandomForest, XGBoost, LightGBM.

Hyperparameter tuning.

Feature selection & importance analysis.

🔹 4.4 Model Evaluation & Testing

Metrics: Accuracy, Precision, Recall, ROC-AUC.

Cross-validation and validation dataset.

Model performance comparison.

🔹 4.5 SHAP Analysis (Explainability)

Feature impact visualization.

Summary and Beeswarm plots.

➤ 5. Frontend & Backend Integration

Backend: Built with FastAPI to serve predictions via an API.

Frontend: Developed using React for user interaction.

Communication: API receives patient data and returns predictions.

➤ 6. Conclusion

This project demonstrates how machine learning can be applied to medical prognosis by combining structured data, advanced predictive models, and explainability techniques. Future improvements include refining the models with additional clinical parameters and expanding the dataset for greater generalizability.

<<<<<<< HEAD
📌 For setup and execution steps, refer to instructions.txt.
=======
<<<<<<< Updated upstream
=======
-------📌 For setup and execution steps, refer to instructions.txt.-------
