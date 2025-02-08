# Advanced Feature Engineering using Pandas

## Introduction
Feature engineering is the process of transforming raw data into meaningful features that enhance the performance of machine learning models. Pandas provides powerful tools for feature creation, transformation, and selection.

---
## 1. Creating New Features
### Sample Data
```python
import pandas as pd

data = {
    "product": ["Laptop", "Phone", "Tablet", "Monitor"],
    "quantity": [2, 5, 3, 4],
    "unit_price": [1000, 500, 300, 200],
    "date_column": pd.to_datetime(["2023-05-10", "2023-06-15", "2023-07-20", "2023-08-25"]),
    "category": ["Electronics", "Electronics", "Electronics", "Accessories"]
}

df = pd.DataFrame(data)
```

### Generating Derived Features
Derived features are created using existing data to provide more insights.
```python
df["total_price"] = df["quantity"] * df["unit_price"]  # Creating a derived column
```

### Extracting Date Components
```python
df["year"] = df["date_column"].dt.year
df["month"] = df["date_column"].dt.month
df["day_of_week"] = df["date_column"].dt.dayofweek
```

---
## 2. Handling Categorical Variables
### One-Hot Encoding
```python
df_encoded = pd.get_dummies(df, columns=["category"], drop_first=True)  # One-hot encoding
```

### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["category"])
```

---
## 3. Feature Scaling and Normalization
### Standardization
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=['number']))
```

### Min-Max Normalization
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df.select_dtypes(include=['number']))
```

---
## 4. Handling Outliers
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
## 5. Feature Selection
### Removing Low Variance Features
```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
df_reduced = selector.fit_transform(df)
```

### Selecting Important Features Using Correlation
```python
correlation_matrix = df.corr()
important_features = correlation_matrix["quantity"].abs().sort_values(ascending=False)
print(important_features)
```

---
## 6. Dimensionality Reduction Techniques
### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df.select_dtypes(include=['number']))
```

### Feature Importance using Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

X = df[["quantity", "unit_price", "total_price"]]
y = df["category_encoded"]
rf = RandomForestClassifier()
rf.fit(X, y)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))
```

---
## 7. Applying Feature Engineering to Real-World Datasets
### Case Study: Predicting Total Price
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[["quantity", "unit_price"]]
y = df["total_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted Total Prices:", predictions)
```

### Case Study: Customer Segmentation
```python
df["total_spent"] = df["quantity"] * df["unit_price"]
```

---
## Conclusion
- Creating new features improves model performance.
- Encoding categorical variables makes data usable for ML models.
- Scaling and normalizing data ensures consistency.
- Handling outliers reduces model bias.
- Selecting relevant features enhances efficiency.
- Dimensionality reduction simplifies complex datasets.
- Real-world datasets benefit significantly from proper feature engineering.

### Next Steps
- Explore **Deep Learning Feature Engineering**
- Implement **Feature Engineering Pipelines**

Happy Coding! 🚀

