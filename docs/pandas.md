# Pandas Guide: Data Analysis Made Easy

## Introduction
Pandas is a powerful Python library for data manipulation and analysis. It provides data structures like **Series** and **DataFrame** that make handling structured data simple and efficient.

### Why Use Pandas?
- **Easier Data Handling**: Works well with structured data like tables and CSVs.
- **Efficient Operations**: Optimized for speed with built-in functions.
- **Seamless Integration**: Works with NumPy, Matplotlib, and other libraries.
- **Industry Applications**: Used in AI/ML, finance, and data science.

---
## 1. Installation & Importing
To install Pandas, run:
```bash
pip install pandas
```
Then, import it in your script:
```python
import pandas as pd
```

---
## 2. Pandas Data Structures
### Series (1D Data)
```python
import pandas as pd

data = [10, 20, 30, 40]
series = pd.Series(data)
print(series)
```

### DataFrame (2D Data)
```python
data = {
    'Name': ['Amresh', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)
print(df)
```

---
## 3. Reading & Writing Data
### Reading CSV
```python
df = pd.read_csv('data.csv')
print(df.head())  # First 5 rows
```

### Writing CSV
```python
df.to_csv('output.csv', index=False)
```

---
## 4. Data Inspection
```python
print(df.info())    # Overview of DataFrame
print(df.describe())  # Summary statistics
print(df.columns)   # List of column names
print(df.shape)     # Rows and columns count
```

---
## 5. Selecting & Filtering Data
```python
print(df['Age'])      # Select a single column
print(df[['Name', 'Salary']])  # Select multiple columns
print(df[df['Age'] > 30])  # Filter rows based on condition
```

---
## 6. Modifying Data
```python
df['Salary'] = df['Salary'] * 1.1  # Increase salaries by 10%
df['New_Column'] = 'Default Value'  # Add new column
df.drop('New_Column', axis=1, inplace=True)  # Remove column
```

---
## 7. Handling Missing Data
```python
df.dropna()  # Remove missing values
df.fillna(0)  # Replace missing values with 0
```

---
## 8. Grouping & Aggregation
```python
grouped = df.groupby('Age')['Salary'].mean()
print(grouped)
```

---
## 9. Merging & Joining DataFrames
```python
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Amresh', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [1, 2, 3], 'Salary': [50000, 60000, 70000]})
merged = pd.merge(df1, df2, on='ID')
print(merged)
```

---
## 10. Time Series Analysis
```python
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(df.resample('M').mean())  # Resample data by month
```

---
## Best Practices
- **Use `astype()` for type conversions**: Convert `object` to `int/float` where needed.
- **Vectorized Operations**: Avoid loops, use built-in Pandas functions for efficiency.
- **Memory Optimization**: Use `float32` instead of `float64` for large datasets.
- **Handle Missing Data Early**: Avoid errors in downstream analysis.

---
## Conclusion
- **Pandas simplifies data manipulation.**
- **Essential for AI/ML, finance, and analytics.**
- **Works seamlessly with NumPy and visualization libraries.**

### Next Steps
- **[Pandas for Data Science](pandas-data-science.md)**
- Explore **Matplotlib & Seaborn** (Data Visualization)
- Learn **Scikit-learn** (Machine Learning)

Happy Learning! 🚀

