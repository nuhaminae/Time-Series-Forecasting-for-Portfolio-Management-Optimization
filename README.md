# Time Series Forecasting for Portfolio Management Optimization

[![CI](https://github.com/nuhaminae/Time-Series-Forecasting-for-Portfolio-Management-Optimization/actions/workflows/CI.yml/badge.svg)](https://github.com/nuhaminae/Time-Series-Forecasting-for-Portfolio-Management-Optimization/actions/workflows/CI.yml)
![Version Control](https://img.shields.io/badge/Artifacts-DVC-brightgreen)
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort Imports](https://img.shields.io/badge/imports-isort-blue.svg)
![Flake8 Lint](https://img.shields.io/badge/lint-flake8-yellow.svg)

## Overview

---

## Key Features

---

## Table of Contents

- [Project Background](#project-background)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [Project Status](#project-status)

---

## Project Background

---

## Data Sources

---

---

## Project Structure

```bash
├── .dvc/                              # Data Version Control
├── .github/                           # CI workflows
├── data/
│   ├── raw/                           # Original datasets
│   └── processed/                     # Cleaned and transformed datasets
├── insights/                          # Plots and charts for reporting
├── notebooks/                         # Notebooks
│   ├── 01_eda.ipynb
|   └── ...
├── scripts/                           # Core scripts
│   ├── __init__.py
|   ├── _01_eda.py
|   └── ...
├── tests/
│   ├── test_01_eda.py
|   ├── test_dummy.py
|   └── ...
├── .dvcignore
├── .flake8
├── .gitignore                         # Ignore unnecessary files
├── .pre-commit-config.yaml            # Pre-commit configuration
├── ...
├── format.ps1                         # Formatting
├── pyproject.toml
├── README.md                          # Project overview and setup instructions
└── requirements.txt                   # Pip install fallback
```

---

## Installation

### Prerequisites

- Python 3.8 or newer (Python 3.11 recommended)
- `pip` (Python package manager)
- [DVC](https://dvc.org/) (for data version control)
- [Git](https://git-scm.com/)

### Setup

```bash
# Clone repo
git clone https://github.com/nuhaminae/Time-Series-Forecasting-for-Portfolio-Management-Optimization
cd Time-Series-Forecasting-for-Portfolio-Management-Optimization
____________________________________________
# Create and activate virtual environment
python -m venv .tslvenv
.tslvenv\Scripts\activate      # On Windows
source .tslvenv/bin/activate   # On Unix/macOS
____________________________________________
# Install dependencies
pip install -r requirements.txt
____________________________________________
# Install and activate pre-commit hooks
pip install pre-commit
pre-commit install
____________________________________________
# (Optional) Pull DVC data
dvc pull
```

---

## Usage

1. **Preprocessing and EDA**
    Run the core preprocessing scripts:

    ```bash
    python scripts/_01_eda.py
    ```

2.

3. **Explore with Notebooks**
    Notebooks are provided for exploratory and iterative development:

    Open with Jupyter or VSCode to navigate the workflow interactively.

4.

5.

6.

7. **Code Quality and Linting**
    This project uses pre-commit hooks to automatically format and lint `.py` and `.ipynb` files using:

    |Tool       | Purpose                                       |
    |:----------|-----------------------------------------------|
    | Black     |Enforces consistent code formatting            |
    | isort     |Sorts and organises import statements          |
    | Flake8    |Lints Python code for style issues             |
    | nbQA      |Runs Black, isort, and Flake8 inside notebooks |

    ``` bash
    # Format and lint all scripts and notebooks
    pre-commit run --all-files
    ```

---

## EDA Visual Insights

---

## Modelling Insights

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.
Make sure to follow best practices for version control, testing, and documentation.

---

## Project Status

The project is underway. Follow the [commit history](https://github.com/nuhaminae/Time-Series-Forecasting-for-Portfolio-Management-Optimization).
