# Python Lambda Functions

Lambda functions in Python are small, anonymous functions that can have any number of arguments but only one expression. They are useful when you need short, throwaway functions without formally defining them using `def`.

## 1. Basic Syntax
The syntax of a lambda function is:
```python
lambda arguments: expression
```
The `expression` is evaluated and returned.

### Example:
```python
add = lambda x, y: x + y
print(add(5, 3))  # Output: 8
```

## 2. Using Lambda with `map()`, `filter()`, and `reduce()`
Lambda functions are often used with higher-order functions like `map()`, `filter()`, and `reduce()`.

### `map()` - Apply a Function to Each Item in a List
```python
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # Output: [1, 4, 9, 16]
```

### `filter()` - Filter Items Based on a Condition
```python
numbers = [1, 2, 3, 4, 5]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # Output: [2, 4]
```

### `reduce()` - Apply a Function Cumulatively (Requires `functools` Module)
```python
from functools import reduce
numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)
print(product)  # Output: 24
```

## 3. Using Lambda in Sorting
Lambda functions can be used with `sorted()` for custom sorting.

### Sorting a List of Tuples
```python
students = [("Amresh", 85), ("Suresh", 90), ("Ganesh", 80)]
sorted_students = sorted(students, key=lambda x: x[1], reverse=True)
print(sorted_students)  # Output: [('Suresh', 90), ('Amresh', 85), ('Ganesh', 80)]
```

## 4. Lambda with Conditional Expressions
A lambda function can include conditions using the ternary operator.
```python
max_number = lambda x, y: x if x > y else y
print(max_number(10, 20))  # Output: 20
```

## 5. Assigning Lambda to Dictionary Keys
Lambda functions can be stored in dictionaries for dynamic function calls.
```python
operations = {
    "add": lambda x, y: x + y,
    "subtract": lambda x, y: x - y,
    "multiply": lambda x, y: x * y
}
print(operations["add"](5, 3))  # Output: 8
```

## 6. Nested Lambda Functions
Lambdas can be nested within other functions.
```python
def multiplier(n):
    return lambda x: x * n

double = multiplier(2)
print(double(5))  # Output: 10
```

## 7. Best Practices and Limitations
### Best Practices:
- Use lambda functions for short, simple operations.
- Prefer `def` functions for complex logic for readability.
- Use with built-in functions like `map()`, `filter()`, and `sorted()`.
- Keep lambda expressions concise to improve code readability.
- Assign lambda functions to variables only when necessary.
- Avoid excessive nesting of lambda functions.
- Use meaningful variable names to enhance clarity.

### Limitations:
- Cannot have multiple expressions.
- Harder to debug and read in complex cases.

Lambda functions provide a concise way to write small, anonymous functions. They are powerful when used appropriately but should be avoided for overly complex logic. 🚀
