# Python Learning Guide

Python is a beginner-friendly programming language known for its simplicity and readability. This guide covers the fundamental concepts of Python step by step with easy-to-understand explanations and examples.

## 1. Introduction to Python
Python is a popular programming language used for web development, data science, automation, and more. It is easy to learn because of its simple syntax.

### Installing Python
To start using Python, you need to install it from [python.org](https://www.python.org/). You can check if Python is installed by running:
```sh
python --version
```

### Running Python Code
You can write Python code in a script (`.py` file) or run it interactively in the terminal.
```python
print("Hello, World!")
```

## 2. Declaring Variables
Variables store data in memory. Python automatically determines the type of data.
```python
x = 10  # Integer
y = 3.14  # Float
z = "Python"  # String
print(x, y, z)
```

## 3. Data Types
Python has different types of data:
```python
integer_value = 10
float_value = 3.14
string_value = "Hello"
boolean_value = True
none_value = None
print(type(integer_value), type(float_value), type(string_value), type(boolean_value), type(none_value))
```

## 4. Lists
Lists are ordered and mutable collections of items.
```python
my_list = [1, 2, 3, 4, 5]
print(my_list)
print(my_list[0])  # Access first element
```

## 5. Tuples
Tuples are like lists but immutable (cannot be changed).
```python
my_tuple = (1, 2, 3, 4, 5)
print(my_tuple)
```

## 6. Dictionaries
Dictionaries store data in key-value pairs.
```python
my_dict = {"name": "Alice", "age": 25}
print(my_dict["name"])
```

## 7. Conditional Statements
Conditional statements help control the flow of the program.
```python
num = 10
if num > 5:
    print("Greater than 5")
else:
    print("5 or less")
```

## 8. Loops
Loops allow repeating actions.
### For Loop
```python
for i in range(5):
    print("Iteration:", i)
```
### While Loop
```python
count = 0
while count < 5:
    print("Count is:", count)
    count += 1
```

## 9. Functions
Functions allow code reuse.
```python
def greet(name):
    return "Hello, " + name
print(greet("Alice"))
```

## 10. Classes and Objects
Object-oriented programming in Python is done using classes.
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"My name is {self.name} and I am {self.age} years old."

p1 = Person("John", 25)
print(p1.introduce())
```

## 11. Importing Libraries
Python has built-in and external libraries for additional functionality.
```python
import math
print(math.sqrt(16))
```

## 12. Working with Files
Python can read and write files.
```python
with open("example.txt", "w") as f:
    f.write("This is a test file.")

with open("example.txt", "r") as f:
    print(f.read())
```

## 13. Exception Handling
Handling errors prevents program crashes.
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
```

## 14. Advanced Topics
- **[Modules and Packages](python-modules-packages.md)**
- **[File Handling with JSON and CSV](python-file-handling.md)**
- **[Regular Expressions](python-regex.md)**
- **[Lambda Functions](python-lambda-function.md)**
- **[List Comprehensions](python-list-comprehensions.md)**
- **[Object-Oriented Programming (OOP) Concepts](python-oops.md)**

This step-by-step guide will help you build a strong foundation in Python. Happy coding! 🚀
