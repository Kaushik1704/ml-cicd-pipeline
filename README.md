# 🚀 CI/CD Pipeline for ML (Fake News Detection)

This project demonstrates how to set up a **CI/CD pipeline** for ML projects using **GitHub Actions**.  
The model is a simple **Fake News Detector** trained with scikit-learn.

---

## 🔹 Features
- Logistic Regression with TF-IDF vectorizer
- Unit tests with pytest
- CI/CD pipeline with GitHub Actions
- Docker container build automation

---

## 🔹 CI/CD Workflow
1. On each push` or pull_request:
   - Run **tests** with pytest
   - Build **Docker image**
2. Ensures code quality before deployment.

---

## 🔹 Run Locally
pip install -r requirements.txt
python src/train.py
python src/inference.py
