# Python Regular Expressions

Regular Expressions (RegEx) in Python allow you to search, match, and manipulate strings efficiently. The `re` module provides built-in support for using regular expressions.

## 1. Importing the `re` Module
Before using regular expressions, import the `re` module:
```python
import re
```

## 2. Basic Pattern Matching
### Searching for a Pattern
Use `re.search()` to find the first occurrence of a pattern.
```python
text = "Hello, my name is Amresh."
pattern = "Amresh"
match = re.search(pattern, text)
if match:
    print("Match found!")
else:
    print("No match found.")
```

### Finding All Matches
Use `re.findall()` to return all occurrences of a pattern.
```python
text = "Amresh, Ganesh, Ramesh, Suresh"
pattern = "esh"
matches = re.findall(pattern, text)
print(matches)  # Output: ['esh', 'esh', 'esh']
```

### Matching at the Start of a String
Use `re.match()` to check if the pattern matches at the beginning of the string.
```python
text = "Amresh loves Python"
pattern = "Amresh"
match = re.match(pattern, text)
if match:
    print("Pattern found at the start!")
```

## 3. Using Special Characters
### Common Special Characters:
- `.` : Matches any character except a newline.
- `^` : Matches the start of the string.
- `$` : Matches the end of the string.
- `*` : Matches zero or more occurrences of the previous character.
- `+` : Matches one or more occurrences of the previous character.
- `?` : Matches zero or one occurrence of the previous character.
- `{n,m}` : Matches between `n` and `m` occurrences.
- `[]` : Matches any one character inside the brackets.
- `|` : Acts as OR (e.g., `cat|dog` matches "cat" or "dog").
- `()` : Groups patterns together.

Example:
```python
text = "apple banana orange Amresh"
pattern = r"b.n.n."  # Matches "banana"
match = re.search(pattern, text)
print(match.group())
```

## 4. Replacing Text with `re.sub()`
Use `re.sub()` to replace occurrences of a pattern in a string.
```python
text = "Amresh loves cats and dogs."
pattern = "cats"
replacement = "birds"
new_text = re.sub(pattern, replacement, text)
print(new_text)  # Output: "Amresh loves birds and dogs."
```

## 5. Splitting a String with `re.split()`
Use `re.split()` to split a string based on a pattern.
```python
text = "Amresh|Ganesh,Ramesh;Suresh"
pattern = r"[,;|]"
result = re.split(pattern, text)
print(result)  # Output: ['Amresh', 'Ganesh', 'Ramesh', 'Suresh']
```

## 6. Compiling Regular Expressions
For efficiency, compile a pattern using `re.compile()`.
```python
pattern = re.compile(r"\d+")  # Matches one or more digits
text = "Amresh has 42 apples and 30 oranges."
matches = pattern.findall(text)
print(matches)  # Output: ['42', '30']
```

## 7. Advanced Concepts
### Using Lookaheads and Lookbehinds
Lookaheads and lookbehinds help match patterns without consuming characters.

#### Positive Lookahead
Matches a pattern only if followed by another pattern.
```python
text = "Amresh123 Python456 Java789"
pattern = r"\w+(?=\d+)"
matches = re.findall(pattern, text)
print(matches)  # Output: ['Amresh', 'Python', 'Java']
```

#### Negative Lookahead
Matches a pattern only if NOT followed by another pattern.
```python
pattern = r"\w+(?!\d+)"
matches = re.findall(pattern, text)
print(matches)  # Output: [] (No matches, as all words are followed by digits)
```

#### Positive Lookbehind
Matches a pattern only if preceded by another pattern.
```python
text = "₹500 $300 €200"
pattern = r"(?<=₹)\d+"
matches = re.findall(pattern, text)
print(matches)  # Output: ['500']
```

#### Negative Lookbehind
Matches a pattern only if NOT preceded by another pattern.
```python
pattern = r"(?<!₹)\d+"
matches = re.findall(pattern, text)
print(matches)  # Output: ['300', '200']
```

## 8. Best Practices
- Use raw strings (`r"pattern"`) to avoid escaping `\`.
- Compile patterns for better performance in repeated use.
- Use `re.VERBOSE` for writing complex regex with comments.
- Test patterns using online regex testers for accuracy.
- Use descriptive variable names for readability.

This guide provides an easy introduction to Python's Regular Expressions, helping you match and manipulate text efficiently! 🚀
