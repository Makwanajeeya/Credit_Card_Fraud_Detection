Credit Card Fraud Detection (Machine Learning Project)
Overview

This project presents a machine learning solution designed to detect fraudulent credit card transactions. The primary focus was on addressing the challenge of highly imbalanced datasets, a common issue in real-world financial fraud detection scenarios. The goal was to build a classification system that reliably minimizes missed fraud cases.

Dataset and Preprocessing

The project used the European card transaction dataset from Kaggle, which contains over 284,000 transactions, of which only 492 are fraudulent (approximately 0.17%).

Imbalance Handling: To ensure the model learned effectively from the minority class, the dataset was balanced using Under-Sampling, creating a robust and equally distributed training environment.

Feature Scaling: The data was preprocessed using RobustScaler to reduce the influence of extreme outliers in the transaction features, ensuring model stability.

Methodology and Results

Multiple classification algorithms were implemented and compared. The Random Forest Classifier was selected for its superior performance and reliability.

Final Model: Random Forest Classifier

Evaluation Focus: Since fraud detection requires minimizing missed fraud cases, the model was evaluated with a focus on recall and the precision-recall trade-off.

Key Metrics Achieved on Test Set:

Accuracy: 0.9340

Recall (Fraud Class): 0.89 (meaning 89% of actual fraudulent cases were successfully detected)

Precision-Recall AUC (PR-AUC): 0.9781 (indicating strong performance across classification thresholds)

Technology Stack

Language: Python

Libraries: Scikit-learn, Pandas, NumPy

Visualization: Matplotlib, Seaborn

Tuning & Deployment: GridSearchCV, joblib (for saving the final model)
