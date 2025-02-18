# AI/ML Beginner's Guide

## 📌 Introduction

This guide provides an overview of AI/ML, from the basics to deploying a model for end-user consumption via a Flask API.

---

## 🔹 Basics of AI/ML

- What is AI?
- What is Machine Learning (ML)?
- Difference between AI, ML, and Deep Learning
- Applications of AI/ML in real life

---

## 🔹 Essential AI/ML Libraries

- **Python** (Primary language for AI/ML development)
- **NumPy, Pandas, Matplotlib** (Data Handling & Visualization)
- **Scikit-Learn** (Traditional ML Algorithms)
- **TensorFlow & PyTorch** (Deep Learning)

---

## 🔹 Types of Machine Learning

1. **Supervised Learning**: Training models on labeled data to make predictions or classifications.

2. **Unsupervised Learning**: Finding patterns and relationships in unlabeled data without predefined categories.

3. **Semi-Supervised Learning**: Combining a small amount of labeled data with a large amount of unlabeled data for training.

4. **Reinforcement Learning**: Training an agent to make decisions by interacting with an environment and receiving rewards or penalties.

5. **Deep Learning**: A subset of machine learning using neural networks with many layers to model complex data representations.

6. **Natural Language Processing (NLP)**: Enabling machines to understand, interpret, and generate human language.

7. **Transfer Learning**: Reusing pre-trained models for new tasks to save time and computational resources.

8. **Ensemble Learning**: Combining multiple models to improve prediction accuracy and performance.

9. **Active Learning**: Actively selecting data points to be labeled by a human to improve model performance with fewer labeled instances.

10. **Online Learning**: Continuously updating models as new data arrives in real-time without retraining from scratch.

For more details on each category, refer to the [detailed documentation](ml-types.md).

---

## 🔹 AI/ML Workflow

1. **Data Collection** - Gather data from various sources.
2. **Data Preprocessing** - Cleaning, transforming, and normalizing data.
3. **Feature Engineering** - Selecting relevant features for the model.
4. **Exploratory Data Analysis (EDA)**
  - Understand data distribution.
  - Identify correlations and trends using visualization tools (Matplotlib, Seaborn).
5. **Model Selection** - Choosing the appropriate algorithm.
6. **Model Training** - Training the model with data.
7. **Model Evaluation** - Checking accuracy, precision, recall, etc.
8. **Hyperparameter Tuning** - Optimizing model performance.
9. **Save the Trained Model** - Storing the trained model for future use.
10. **Model Deployment** - Making the model available for users.

---

## 🔹 Save the Trained Model

After training a model, it is crucial to save it for future use.

### **Using Pickle to Save and Load Models**

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

---

## 🔹 Model Deployment

- Deploying models using Flask API.
- Using FastAPI for faster responses.
- Deploying to cloud platforms (AWS, Google Cloud, Azure).

---

## 🔹 Using Deployed Model in Flask API

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

