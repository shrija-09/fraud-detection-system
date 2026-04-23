# 💳 Real-Time Fraud Detection System

## 📌 Overview
Machine learning system to detect fraudulent transactions with real-time API support.

## 🚀 Features
- Random Forest model (ROC-AUC: 0.95+)
- FastAPI-based prediction endpoint
- Clean modular structure

## 🛠️ Tech Stack
Python, Scikit-learn, FastAPI, Pandas, NumPy

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python model/train.py
uvicorn api.main:app --reload
