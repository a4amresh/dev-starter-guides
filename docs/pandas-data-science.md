# Pandas for Data Science: Data Cleaning & Processing

## Introduction
Pandas is a powerful Python library for data manipulation and analysis, widely used in data science. This guide covers essential data cleaning techniques, correlations, and best practices using Pandas.

---
## 1. Handling Missing Data (Empty Cells)
### Checking for Missing Data
```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.isnull().sum())  # Count missing values in each column
```

### Filling Missing Data
```python
df.fillna("Unknown", inplace=True)  # Replace missing values with 'Unknown'
df.fillna(df.mean(), inplace=True)  # Replace missing values with column mean
```

### Dropping Missing Data
```python
df.dropna(inplace=True)  # Remove rows with missing values
```

---
## 2. Handling Incorrect Data Format
### Converting Data Types
```python
df["date"] = pd.to_datetime(df["date"])  # Convert to datetime
df["price"] = df["price"].astype(float)  # Convert to float
```

---
## 3. Handling Incorrect Data Entries
### Identifying Incorrect Data
```python
print(df["age"].unique())  # Find unique values to check anomalies
```

### Fixing Incorrect Data
```python
df.loc[df["age"] < 0, "age"] = df["age"].median()  # Replace negative ages with median
```

---
## 4. Removing Duplicates
### Finding Duplicates
```python
print(df.duplicated().sum())  # Count duplicate rows
```

### Removing Duplicates
```python
df.drop_duplicates(inplace=True)
```

---
## 5. Data Correlation with Pandas
Correlation helps in understanding relationships between variables.

### Checking Correlation
```python
correlation_matrix = df.corr()
print(correlation_matrix)
```

### Visualizing Correlation
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()
```

---
## 6. Data Transformation & Normalization
Transforming data helps improve model performance and accuracy.

### Standardizing Data
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=['number']))
```

### Normalizing Data
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df.select_dtypes(include=['number']))
```

---
## 7. Handling Outliers
### Detecting Outliers Using IQR (Interquartile Range)
```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
print(outliers.sum())  # Count of outliers in each column
```

### Removing Outliers
```python
df = df[~outliers.any(axis=1)]
```

---
## 8. Feature Engineering with Pandas
### Creating New Features
```python
df["total_price"] = df["quantity"] * df["unit_price"]  # Creating a derived column
```

### Encoding Categorical Variables
```python
encoded_df = pd.get_dummies(df, columns=["category"], drop_first=True)  # One-hot encoding
```

---
## Conclusion
- Handling missing values improves dataset quality.
- Converting incorrect formats ensures consistency.
- Removing duplicates prevents data bias.
- Correlation helps understand data relationships.
- Transformation and normalization prepare data for analysis.
- Feature engineering creates valuable new insights.

### Next Steps
- Learn **[Advanced Feature Engineering](pandas-advanced-feature-engineering.md)** using Pandas
- Explore **[Machine Learning with Cleaned Data](pandas-ml-with-cleaned-data.md)**

Happy Coding! 🚀
