💳 Credit Card Fraud Detection using Machine Learning

📌 Project Overview

This project aims to detect fraudulent credit card transactions using Machine Learning models. The system compares Logistic Regression and Decision Tree classifiers to identify fraud transactions accurately.
Fraud detection is important to reduce financial losses and improve transaction security.

📂 Dataset

Dataset Name: Credit Card Fraud Detection
Source: Kaggle
Features: Transaction details (V1–V28, Amount, Time)
Target Column: Class
0 → Normal Transaction
1 → Fraud Transaction
The dataset is highly imbalanced (very few fraud cases).

⚙️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

🧠 Machine Learning Models Used
Logistic Regression
Decision Tree Classifier
🔄 Project Workflow
Data Loading
Data Preprocessing
Train-test split
Feature scaling using StandardScaler
Model Training
Model Evaluation
Accuracy
Confusion Matrix
Classification Report
Model Comparison Visualization

📊 Results

Both models were evaluated based on accuracy and classification metrics.
Logistic Regression performs well for linear patterns, while Decision Tree captures complex relationships.
Since the dataset is imbalanced, metrics like Precision, Recall, and F1-score are more important than accuracy.

🚀 Future Improvements

Use Random Forest or XGBoost
Apply SMOTE for handling imbalanced data
Hyperparameter tuning
Deploy the model using Flask or Streamlit

▶️ How to Run the Project

Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn
Place creditcard.csv in the project folder.
Run the Python script:
python fraud_detection.py


📌 Conclusion

The project demonstrates how machine learning can effectively detect fraudulent transactions. With proper preprocessing and model tuning, fraud detection systems can significantly improve financial security.
