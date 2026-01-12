# Fraud Detection using Anomaly Detection Models

This project focuses on detecting fraudulent transactions using unsupervised anomaly detection techniques and deploying the final model as a Python API.

---

## Project Overview

Fraud detection is a highly imbalanced problem where fraudulent transactions are rare but costly. Instead of using supervised classification, this project
explores unsupervised anomaly detection models that learn normal transaction behavior and flag unusual patterns as potential fraud.

---

## Dataset Description

A synthetic transaction dataset was created to show realistic fraud behaviour instead of random labeling. Fraud probability is influenced by multiple factors
such as transaction amount, time, distance from home, and merchant category, with added noise to avoid rule-based separation.

Key characteristics:
- 100,000 transactions
- Fraud rate below 2%
- Strong class imbalance
- Behavioral and numerical features

---

## Models Implemented

The following models were implemented and evaluated:

- **Isolation Forest**  
  Used as the primary anomaly detection model due to its speed, scalability, and minimum preprocessing requirements.

- **One-Class SVM**  
  Tested as an alternative but found to be slower and more sensitive to feature scaling.

- **Autoencoder**  
  Neural Network model used for comparison, but considered harder to deploy due to higher inference complexity.

---

## Evaluation Metrics

Because accuracy is misleading for imbalanced fraud data, the following metrics were used:

- Precision
- Recall
- F1-score
- ROC AUC (using anomaly scores)

More emphasis was given on precision because false positives are costly in real-world payment systems.

---

## API Deployment

The final Isolation Forest model was deployed as a FastAPI-based Python service with a `/predict` endpoint.

Features of the API:
- Request validation using Pydantic
- Error handling
- Logging for predictions
- Low-latency inference suitable for real-time use

The trained model is loaded from a serialized `.pkl` file for efficient reuse.

---

## Project Structure

## Project Structure

```text
fraud_detection_assignment/
├── data/
│   └── transactions.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_isolation_forest.ipynb
│   ├── 03_one_class_svm.ipynb
│   └── 04_autoencoder.ipynb
│
├── models/
│   └── isolation_forest.pkl
│
├── api/
│   └── app.py
│
├── reports/
│
└── README.md

---

## How to Run the Project

1. Open the notebooks in order to review data creation and model experiments.
2. To start the API locally, run:
python -m uvicorn api.app:app --reload
3. Open the Browser and navigate to: http://127.0.0.1:8000/docs


