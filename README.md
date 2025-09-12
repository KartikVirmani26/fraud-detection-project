Overview
This project implements a simple machine learning pipeline to detect fraudulent transactions using synthetic data. Class imbalance is tackled with the SMOTE technique, and a Random Forest classifier is trained and evaluated using essential metrics.
Features
•	Synthetic data generation for normal and fraudulent transactions
•	Proper train-test split method and use of SMOTE to balance class distributions
•	Random Forest model training
•	Key evaluation metrics: accuracy, confusion matrix, precision, recall, F1-score
Setup
1.	Clone the repository or download the files.
2.	Install dependencies using pip (recommended in a virtual environment):
pip install -r requirements.txt

Requirements:
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib

How to Run
Run the fraud detection script from the command line:
python fraud_detection.py

Output Example
Sample output:
Dataset: 5000 samples, 22 features
Normal: 4853, Fraud: 147

After SMOTE: 6794 balanced samples
Training Random Forest...

=== Results ===
Accuracy: 0.970 (97.0%)

Confusion Matrix:
           Predicted
        Normal  Fraud
Normal    1408    13
Fraud       48    31

Fraud Detection Metrics:
Precision: 0.705 (70.5%)
Recall:    0.646 (64.6%)
F1-Score:  0.675 (67.5%)
Frauds detected: 31/79
False alarms: 13

✓ Fraud detection completed!

You can also add a screenshot or image here, e.g.:




 
Project Structure
fraud-detection-project/
├── fraud_detection.py
├── requirements.txt
├── README.md
└── screenshots/
    ├── output_example.png
    └── confusion_matrix.png
