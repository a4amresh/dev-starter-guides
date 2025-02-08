# NumPy Guide

## Introduction
NumPy (Numerical Python) is a powerful library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate efficiently on these arrays.

### Why Use NumPy?
- **Faster than Python lists** (optimized with C).
- **Memory-efficient** (compact storage).
- **Supports mathematical operations** like linear algebra, statistics, and random numbers.
- **Widely used in AI/ML, Data Science, Finance, and scientific computing.**

---
## 1. Installation & Importing
To install NumPy, run:
```bash
pip install numpy
```
Then, import it in your script:
```python
import numpy as np
```

---
## 2. Creating Arrays

### 1D Array (Vector)
**Example: Stock Prices Over a Week**
```python
stock_prices = np.array([150, 152, 153, 149, 148, 151, 154])
print(stock_prices)
```

### 2D Array (Matrix)
**Example: Monthly Sales Data (Rows: Products, Columns: Months)**
```python
sales_data = np.array([[200, 220, 250], [180, 190, 210], [150, 170, 200]])
print(sales_data)
```

### 3D Array
**Example: Image Representation (Height, Width, RGB Channels)**
```python
image = np.random.randint(0, 256, (128, 128, 3))  # Random 128x128 RGB Image
print(image.shape)
```

---
## 3. Array Properties
```python
print(arr.ndim)    # Number of dimensions
print(arr.shape)   # Shape of the array
print(arr.size)    # Total number of elements
print(arr.dtype)   # Data type of elements
```

---
## 4. Creating Special Arrays

### Zeros and Ones
```python
np.zeros((2, 3))  # 2x3 matrix of zeros
np.ones((3, 3))   # 3x3 matrix of ones
```

### Random Numbers
**Example: Generating Synthetic Customer Ratings (1-5)**
```python
customer_ratings = np.random.randint(1, 6, (5, 5))
print(customer_ratings)
```

### Identity Matrix
```python
np.eye(4)  # 4x4 Identity matrix
```

---
## 5. Indexing & Slicing

### Accessing Elements
```python
arr[1]   # 2nd element
```

### Slicing Arrays
```python
arr[1:4]    # Elements 2 to 4
arr[:3]     # First 3 elements
arr[::2]    # Every second element
```

### Accessing 2D Arrays
```python
arr2d[1, 2]   # Row 1, Column 2
arr2d[:, 1]   # All rows, Column 1
```

---
## 6. Mathematical Operations

### Element-wise Operations
**Example: Adjusting Employee Salaries by a 5% Increase**
```python
salaries = np.array([50000, 55000, 60000])
increased_salaries = salaries * 1.05
print(increased_salaries)
```

### Universal Functions
```python
np.sqrt(a)   # Square root
np.exp(a)    # Exponential
np.log(a)    # Natural log
np.sum(a)    # Sum of elements
np.mean(a)   # Mean
```

---
## 7. Reshaping & Transposing

### Reshaping
```python
arr.reshape((2, 3))
```

### Transposing
**Example: Transposing Sales Data to View Monthly Trends**
```python
print(sales_data.T)  # Swaps rows and columns
```

---
## 8. Stacking & Splitting

### Stacking Arrays
```python
np.vstack((a, b))  # Vertical stack
np.hstack((a, b))  # Horizontal stack
```

### Splitting Arrays
```python
np.split(arr, 2)   # Split into 2 parts
```

---
## 9. Boolean Indexing
**Example: Filtering Customers with High Ratings**
```python
high_ratings = customer_ratings[customer_ratings >= 4]
print(high_ratings)
```

---
## 10. Advanced NumPy

### Broadcasting
```python
arr + 10  # Adds 10 to all elements
```

### Linear Algebra
**Example: Solving a System of Linear Equations**
```python
A = np.array([[2, 3], [1, 2]])
b = np.array([8, 5])
x = np.linalg.solve(A, b)
print(x)
```

### Statistical Functions
```python
np.std(arr)  # Standard deviation
np.var(arr)  # Variance
np.median(arr)  # Median
```

### Saving & Loading Data
```python
np.save("array.npy", arr)  # Save array
np.load("array.npy")  # Load array
```

---
## Best Practices
- **Use `astype()` for type conversions**: Convert `int` to `float` when necessary to avoid unintended integer division.
- **Use vectorized operations**: Avoid loops by using NumPy’s built-in functions.
- **Leverage broadcasting**: Enables operations between arrays of different shapes efficiently.
- **Optimize memory usage**: Use `np.float32` instead of `np.float64` for large datasets when high precision isn't needed.

---
## Conclusion
- **NumPy is essential for numerical computing.**
- **Provides fast operations on arrays.**
- **Used in AI/ML, data science, finance, and engineering fields.**

### Next Steps
- Explore **Pandas** (Data Analysis)
- Learn **Matplotlib** (Visualization)
- Dive into **[Machine Learning with NumPy](numpy-ml.md)**

Happy Learning! 🚀
