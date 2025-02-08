# Machine Learning with Cleaned Data

## Introduction
Once data is cleaned and preprocessed, it becomes ideal for building robust machine learning models. This guide demonstrates a typical ML pipeline using cleaned data—from loading and exploring data to model building and evaluation—using Python's Pandas and scikit-learn libraries.

---

## 1. Loading Cleaned Data
Assuming the data has been cleaned and stored in a CSV file, we can load it using Pandas. Alternatively, we can simulate a cleaned dataset with sample data.

### Loading from a CSV File
```python
import pandas as pd

# Load the cleaned data
# Make sure 'cleaned_data.csv' exists in your working directory
df = pd.read_csv('cleaned_data.csv')
print(df.head())
```

### Using Sample Data
```python
import pandas as pd

# Sample cleaned data
data = {
    'quantity': [2, 5, 3, 4, 6],
    'unit_price': [1000, 500, 300, 200, 800],
    'total_price': [2000, 2500, 900, 800, 4800]  # Derived as quantity * unit_price
}
df = pd.DataFrame(data)
print(df.head())
```

---

## 2. Exploratory Data Analysis (EDA)
Perform initial data exploration to understand distributions and relationships in the dataset.

```python
# Descriptive statistics
print(df.describe())

# Data information (types, non-null counts)
print(df.info())

# Check correlations between variables
correlation_matrix = df.corr()
print(correlation_matrix)
```

---

## 3. Splitting Data into Training and Testing Sets
To evaluate model performance, we split the dataset into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Define features and target variable
X = df[['quantity', 'unit_price']]
y = df['total_price']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 4. Model Building: Linear Regression
We build a linear regression model to predict the total price based on quantity and unit price.

```python
from sklearn.linear_model import LinearRegression

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
print('Predicted Total Prices:', predictions)
```

---

## 5. Model Evaluation
Evaluate the model using metrics such as Mean Squared Error (MSE) and R-squared to understand performance.

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print('Mean Squared Error:', mse)
print('R-squared:', r2)
```

---

## 6. Conclusion
In this guide, we demonstrated how to build a machine learning model using cleaned data:

- **Data Loading**: Importing and inspecting the cleaned dataset using Pandas.
- **Exploratory Data Analysis (EDA)**: Understanding data through descriptive statistics and correlations.
- **Data Splitting**: Dividing data into training and testing sets.
- **Model Building**: Training a linear regression model to predict a target variable.
- **Model Evaluation**: Assessing model performance using MSE and R-squared.

### Next Steps
- Experiment with other machine learning algorithms (e.g., decision trees, ensemble methods).
- Implement cross-validation for a more robust evaluation.
- Explore hyperparameter tuning to optimize model performance.

Happy Learning! 🚀

