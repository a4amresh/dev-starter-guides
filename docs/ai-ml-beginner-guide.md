# AI/ML Beginner's Guide

## 📌 Introduction

This guide provides an overview of AI/ML, from the basics to deploying a model for end-user consumption via a Flask API.

---

## 🔹 1️. Basics of AI/ML

- What is AI?
- What is Machine Learning (ML)?
- Difference between AI, ML, and Deep Learning
- Applications of AI/ML in real life

---

## 🔹 2. Essential AI/ML Libraries

- **Python** (Primary language for AI/ML development)
- **NumPy, Pandas, Matplotlib** (Data Handling & Visualization)
- **Scikit-Learn** (Traditional ML Algorithms)
- **TensorFlow & PyTorch** (Deep Learning)

---

## 🔹 3️. Types of Machine Learning

### 1. Supervised Learning

Supervised learning is a type of ML where the model is trained on labeled data.

- **Classification** (e.g., Spam Detection, Image Recognition)
  - Logistic Regression
  - Decision Trees
  - Support Vector Machines (SVM)
  - Neural Networks
- **Regression** (e.g., House Price Prediction, Sales Forecasting)
  - Linear Regression
  - Random Forest Regression
  - Gradient Boosting Machines

### 2. Unsupervised Learning

Unsupervised learning deals with unlabeled data and finds hidden patterns in the dataset.

- **Clustering** (e.g., Customer Segmentation, Anomaly Detection)
  - K-Means Clustering
  - DBSCAN
  - Hierarchical Clustering
- **Association** (e.g., Market Basket Analysis)
  - Apriori Algorithm
  - FP-Growth Algorithm

### 3. Semi-Supervised Learning

Semi-Supervised Learning is a combination of Supervised and Unsupervised Learning. It is useful when labeled data is limited and expensive to obtain.

- **Examples:** Web page classification, Fraud Detection
- **Algorithms:**
  - Self-training
  - Label Propagation
  - Generative Adversarial Networks (GANs)

### 4. Reinforcement Learning

Reinforcement Learning (RL) is based on an agent that interacts with an environment to maximize rewards.

- **Policy-based Learning** (e.g., Policy Gradient, Actor-Critic)
- **Value-based Learning** (e.g., Q-Learning, Deep Q Networks)
- **Model-free and Model-based approaches**
- **Examples:** Robotics, Game Playing (e.g., AlphaGo, Chess AI)

---

## 🔹 4️. Key Steps in ML Workflow

### 1. Data Collection & Preparation
- Gather data from various sources
- Clean and preprocess data (handling missing values, normalization).

### 2. Exploratory Data Analysis (EDA)
- Understand data distribution.
- Identify correlations and trends using visualization tools (Matplotlib, Seaborn).

### 3. Feature Engineering
- Select important features.
- Transform raw data into meaningful input.

### 4. Model Selection & Training
- Choose an appropriate algorithm.
- Split data into training and test sets.
- Train the model using frameworks (Scikit-learn, TensorFlow, PyTorch).

### 5. Model Evaluation
- Use metrics like Accuracy, Precision, Recall, F1-score.
- Cross-validation to test generalization.

### 6. Hyperparameter Tuning
- Optimize model performance using GridSearchCV, RandomizedSearchCV.

### 7. Save the Trained Model
After training a model, it is crucial to save it for future use.

> **Using Pickle to Save and Load Models**

1. **Save the model:**
```python
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
```
2. **Load the model:**
```python
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
```


### 8. Model Deployment
- Convert model to a deployable format (.pkl, .h5, .onnx).
- Deploying models using Flask API
- Using FastAPI for faster responses
- Deploying to cloud platforms (AWS, Google Cloud, Azure)

---

## 🔹 5️. Using Deployed Model in Flask API

1. **Install Flask**
   ```sh
   pip install flask
   ```
2. **Create a Flask app**
   ```python
   from flask import Flask, request, jsonify
   import pickle
   import numpy as np

   app = Flask(__name__)
   model = pickle.load(open('model.pkl', 'rb'))

   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.get_json()
       prediction = model.predict(np.array(data['features']).reshape(1, -1))
       return jsonify({'prediction': prediction.tolist()})

   if __name__ == '__main__':
       app.run(debug=True)
   ```
3. **Run Flask API**
   ```sh
   python app.py
   ```
4. **Send a request using Postman or Curl**
   ```sh
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
   ```

---

## Next Steps
- **[AI/ML Model Creation and Deployment](ai-ml-model-creation-deployment.md)**

## 🎯 Conclusion

This guide covers AI/ML from the basics to deploying models using Flask. Keep practicing and experimenting with different datasets and algorithms! 🚀

