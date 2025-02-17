# Using Python on macOS with Virtual Environments (venv)

## 1. Installing Python on macOS
macOS comes with a pre-installed version of Python, but it is recommended to install the latest version manually to avoid conflicts.

### Install Python via Homebrew (Recommended)
1. Open Terminal.
2. Install Homebrew if not already installed:
   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Install Python:
   ```sh
   brew install python
   ```
4. Verify installation:
   ```sh
   python3 --version
   ```

## 2. Setting Up a Virtual Environment (venv)
A virtual environment helps isolate Python dependencies per project.

### Create a Virtual Environment
1. Navigate to your project directory:
   ```sh
   cd /path/to/your/project
   ```
2. Create a virtual environment:
   ```sh
   python3 -m venv venv
   ```
   This will create a folder named `venv` inside your project directory.

### Activate the Virtual Environment
- **For macOS/Linux:**
  ```sh
  source venv/bin/activate
  ```
- **For Windows (if using on a different machine):**
  ```sh
  venv\Scripts\activate
  ```

You should see `(venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

### Installing Packages Inside Virtual Environment
Once the virtual environment is activated, use `pip` to install dependencies:
```sh
pip install package_name
```
For example:
```sh
pip install numpy pandas
```

### Deactivating the Virtual Environment
To exit the virtual environment:
```sh
deactivate
```

## 3. Removing a Virtual Environment
If you no longer need a virtual environment, you can delete the `venv` folder:
```sh
rm -rf venv
```

## 4. Additional Tips
- To make `python` default to `python3`, add this to your `~/.zshrc` or `~/.bashrc`:
  ```sh
  alias python=python3
  ```
- Upgrade `pip` inside the virtual environment:
  ```sh
  pip install --upgrade pip
  ```
- To check installed packages:
  ```sh
  pip list
  ```
- To freeze dependencies into a requirements file:
  ```sh
  pip freeze > requirements.txt
  ```
- To install dependencies from a requirements file:
  ```sh
  pip install -r requirements.txt
  ```

Now you are ready to use Python and virtual environments effectively on macOS!

