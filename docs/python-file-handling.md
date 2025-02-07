# Python File Handling with JSON and CSV

Python provides built-in support for handling files, including working with JSON and CSV formats. These formats are widely used for storing and exchanging data.

## 1. Handling Text Files
### Opening a File
Use the `open()` function to read or write files.
```python
# Opening a file in read mode
with open("example.txt", "r") as file:
    content = file.read()
    print(content)
```

### Writing to a File
```python
# Opening a file in write mode
with open("example.txt", "w") as file:
    file.write("Hello, World!\n")
```

### Appending to a File
```python
# Opening a file in append mode
with open("example.txt", "a") as file:
    file.write("Appending new content.\n")
```

## 2. Handling JSON Files
### Importing JSON Module
```python
import json
```

### Writing JSON Data to a File
```python
data = {"name": "Alice", "age": 25, "city": "New York"}
with open("data.json", "w") as file:
    json.dump(data, file, indent=4)
```

### Reading JSON Data from a File
```python
with open("data.json", "r") as file:
    data = json.load(file)
    print(data)
```

### Converting JSON to String
```python
json_string = json.dumps(data, indent=4)
print(json_string)
```

## 3. Handling CSV Files
### Importing CSV Module
```python
import csv
```

### Writing to a CSV File
```python
data = [["Name", "Age", "City"], ["Alice", 25, "New York"], ["Bob", 30, "Los Angeles"]]
with open("data.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
```

### Reading from a CSV File
```python
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
```

### Writing to a CSV File Using Dictionary
```python
data = [{"Name": "Alice", "Age": 25, "City": "New York"}, {"Name": "Bob", "Age": 30, "City": "Los Angeles"}]
with open("data_dict.csv", "w", newline='') as file:
    fieldnames = ["Name", "Age", "City"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
```

### Reading from a CSV File Using Dictionary
```python
with open("data_dict.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)
```

## 4. Best Practices
- Always use `with` statement for file handling to ensure proper closure.
- Use `json.dump()` and `json.load()` for working with JSON files.
- Use `csv.writer()`, `csv.DictWriter()`, `csv.reader()`, and `csv.DictReader()` for better handling of CSV data.
- Handle exceptions using `try-except` blocks to avoid crashes.

This guide provides a step-by-step approach to handling files in Python, covering text, JSON, and CSV formats efficiently. 🚀
