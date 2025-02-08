# Machine Learning with NumPy

## Introduction
NumPy is a fundamental library for numerical computing in Python and serves as the backbone for many Machine Learning (ML) libraries. Understanding how to use NumPy for ML tasks helps in building efficient models, handling large datasets, and performing matrix operations essential for AI/ML applications.

### Why Use NumPy for ML?
- **Optimized for performance** (built in C, supports vectorized operations).
- **Handles large datasets efficiently** (memory-efficient storage and operations).
- **Essential for data preprocessing, linear algebra, and statistical analysis**.
- **Used in financial modeling, risk analysis, and AI-powered trading.**

---
## 1. Setting Up NumPy for ML
### Installation
```bash
pip install numpy
```
### Importing NumPy
```python
import numpy as np
```

---
## 2. NumPy for Data Handling

### Creating and Managing Data
**Example: Stock Market Prices**
```python
stock_prices = np.array([100, 102, 105, 110, 108, 107])
print(stock_prices)
```

### Handling Missing Data
**Example: Filling Missing Financial Data**
```python
prices = np.array([100, np.nan, 105, 110, np.nan, 107])
filled_prices = np.nan_to_num(prices, nan=np.mean(prices[~np.isnan(prices)]))
print(filled_prices)
```

---
## 3. Data Preprocessing

### Normalization & Standardization
**Example: Standardizing Stock Prices**
```python
mean = np.mean(stock_prices)
std_dev = np.std(stock_prices)
standardized_prices = (stock_prices - mean) / std_dev
print(standardized_prices)
```

### Feature Scaling
**Example: Min-Max Scaling for Investment Portfolio**
```python
portfolio_returns = np.array([5, 10, 15, 20, 25])
scaled_returns = (portfolio_returns - np.min(portfolio_returns)) / (np.max(portfolio_returns) - np.min(portfolio_returns))
print(scaled_returns)
```

---
## 4. Matrix Operations for ML

### Matrix Representation
**Example: Stock Price Correlation Matrix**
```python
stocks = np.array([[100, 102, 105], [98, 100, 102], [105, 107, 110]])
corr_matrix = np.corrcoef(stocks)
print(corr_matrix)
```

### Solving Linear Equations (Regression Basics)
**Example: Predicting Stock Prices Based on Factors**
```python
A = np.array([[1, 200], [1, 220], [1, 250]])  # [Intercept, Feature]
b = np.array([50, 55, 65])  # Target values
weights = np.linalg.lstsq(A, b, rcond=None)[0]
print(weights)
```

---
## 5. Probability & Statistics in ML

### Mean, Variance, and Standard Deviation
**Example: Measuring Volatility in Stock Prices**
```python
volatility = np.std(stock_prices)
print(volatility)
```

### Generating Random Data (Monte Carlo Simulations)
**Example: Simulating Future Stock Prices**
```python
simulated_prices = stock_prices[-1] + np.random.randn(1000) * volatility
print(simulated_prices[:10])
```

---
## 6. Implementing ML Algorithms with NumPy

### Linear Regression
**Example: Predicting House Prices Based on Features**
```python
X = np.array([[1, 1400], [1, 1600], [1, 1800], [1, 2000]])  # Features (Intercept, Size)
y = np.array([245000, 275000, 305000, 335000])  # Prices
weights = np.linalg.inv(X.T @ X) @ X.T @ y  # Normal Equation
print(weights)
```

### Logistic Regression (Binary Classification)
**Example: Predicting Loan Default (1 = Default, 0 = No Default)**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[1, 700], [1, 800], [1, 900], [1, 1000]])  # Features (Intercept, Credit Score)
y = np.array([0, 0, 1, 1])  # Loan Default Outcome
weights = np.random.rand(2)  # Initialize weights

z = np.dot(X, weights)
predictions = sigmoid(z)
print(predictions)
```

---
## 7. Principal Component Analysis (PCA) for Dimensionality Reduction

### PCA Implementation
**Example: Reducing Dimensions of Financial Data**
```python
from numpy.linalg import eig

cov_matrix = np.cov(stocks.T)
eigenvalues, eigenvectors = eig(cov_matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)
```

---
## 8. Best Practices for Using NumPy in ML

- **Use `astype()` for data type conversions** to avoid unintended errors.
- **Leverage vectorized operations** for faster computations (avoid loops).
- **Use broadcasting** to perform operations efficiently on different-sized arrays.
- **Optimize memory usage** by using lower precision data types (e.g., `np.float32` instead of `np.float64`).

---
## Conclusion
- **NumPy is a core component of Machine Learning workflows.**
- **It enables fast computations and efficient data handling.**
- **Used for data preprocessing, mathematical operations, and implementing ML algorithms.**

### Next Steps
- Explore **Pandas** for Data Analysis
- Learn **Scikit-learn** for ML Model Implementation
- Practice **Deep Learning with TensorFlow & PyTorch**

Happy Learning! 🚀
