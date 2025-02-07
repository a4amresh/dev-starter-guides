# Python Object-Oriented Programming (OOP) Concepts

Object-Oriented Programming (OOP) is a programming paradigm that organizes code into objects, promoting modularity and reusability.

## 1. Key OOP Concepts
### 1.1 Class and Object
- A **class** is a blueprint for creating objects.
- An **object** is an instance of a class.

#### Example:
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, my name is {self.name}."

# Creating an object
amresh = Person("Amresh", 25)
print(amresh.greet())  # Output: Hello, my name is Amresh.
```

### 1.2 Encapsulation
Encapsulation restricts direct access to variables and allows controlled modification using getters and setters.

#### Example:
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private attribute
    
    def deposit(self, amount):
        self.__balance += amount
    
    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())  # Output: 1500
```

### 1.3 Inheritance
Inheritance allows a class to inherit attributes and methods from another class.

#### Example:
```python
class Animal:
    def speak(self):
        return "Animal speaks"

class Dog(Animal):
    def speak(self):
        return "Bark!"

dog = Dog()
print(dog.speak())  # Output: Bark!
```

### 1.4 Polymorphism
Polymorphism allows the same interface to be used for different types.

#### Example:
```python
class Bird:
    def sound(self):
        return "Chirp"

class Cat:
    def sound(self):
        return "Meow"

animals = [Bird(), Cat()]
for animal in animals:
    print(animal.sound())
# Output: Chirp
#         Meow
```

### 1.5 Abstraction
Abstraction hides implementation details and exposes only essential features.

#### Example:
```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def move(self):
        pass

class Car(Vehicle):
    def move(self):
        return "Car is moving"

car = Car()
print(car.move())  # Output: Car is moving
```

## 2. Advanced OOP Concepts
### 2.1 Multiple Inheritance
A class can inherit from multiple parent classes.

#### Example:
```python
class A:
    def method_a(self):
        return "Method A"

class B:
    def method_b(self):
        return "Method B"

class C(A, B):
    pass

obj = C()
print(obj.method_a())  # Output: Method A
print(obj.method_b())  # Output: Method B
```

### 2.2 Method Overriding
A subclass can override a method from its parent class.

#### Example:
```python
class Parent:
    def show(self):
        return "Parent Method"

class Child(Parent):
    def show(self):
        return "Child Method"

obj = Child()
print(obj.show())  # Output: Child Method
```

### 2.3 Super Function
The `super()` function allows calling methods from a parent class inside a child class.

#### Example:
```python
class Parent:
    def greet(self):
        return "Hello from Parent"

class Child(Parent):
    def greet(self):
        return super().greet() + " and Hello from Child"

obj = Child()
print(obj.greet())  # Output: Hello from Parent and Hello from Child
```

### 2.4 Operator Overloading
Operator overloading allows defining custom behavior for operators in user-defined classes.

#### Example:
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(2, 3)
v2 = Vector(4, 5)
v3 = v1 + v2  # Calls __add__
print(v3.x, v3.y)  # Output: 6 8
```

## 3. Best Practices in OOP
- Use meaningful class and method names.
- Follow the **Single Responsibility Principle (SRP)** – each class should have a single purpose.
- Use **encapsulation** to protect class attributes.
- Apply **inheritance** judiciously to avoid deep hierarchy.
- Leverage **polymorphism** to write flexible code.
- Use **abstract classes** for enforcing structure.
- Avoid **multiple inheritance** unless necessary to prevent complexity.
- Utilize **`super()`** to enhance code reusability.

OOP in Python helps in writing organized, maintainable, and reusable code, making it a powerful paradigm for software development! 🚀
