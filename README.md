[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

#  Breast Cancer Prediction Using Machine Learning

A complete end-to-end Machine Learning project for predicting whether a breast tumor is **benign** or **malignant**, using EDA, preprocessing, ML models, evaluation metrics, hyperparameter tuning, and an optimized SVM classifier.

---

##  Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features Implemented](#features-implemented)
- [Technologies Used](#technologies-used)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Best Model](#best-model)
- [License](#license)
- [Author](#author)

---

##  Overview
This project implements a **complete Machine Learning workflow** for breast cancer classification, including:

- Data loading & cleaning  
- Exploratory Data Analysis (EDA)  
- Train–test split  
- Feature scaling  
- Training **six ML models**  
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC  
- Hyperparameter tuning using GridSearchCV  
- Final optimized **Support Vector Machine (SVM)** classifier  

---

##  Dataset
**Breast Cancer Wisconsin Diagnostic Dataset**  
Loaded directly with:

```python
from sklearn.datasets import load_breast_cancer
````

* **569 samples**
* **30 numerical features**
* **Binary target:**

  * `0 = Malignant`
  * `1 = Benign`

 No external download required.

---

##  Features Implemented

* Full preprocessing pipeline
* Exploratory Data Analysis (EDA)
* Machine Learning models:

  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
  * KNN
  * Naive Bayes
  * Artificial Neural Network (ANN)
* Model comparison
* Confusion matrices
* ROC-AUC curves
* Hyperparameter tuning
* Final optimized SVM model

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab / Jupyter Notebook

---

##  Repository Structure

```
Breast-Cancer-Classification
│
├── CCP_Report.pdf                 # Full written report
├── Breast_Cancer_Project.ipynb    # Complete implementation code
├── README.md                      # Documentation
├── LICENSE                        # MIT License
└── requirements.txt               # Python dependencies
```

---

##  Installation

```bash
git clone https://github.com/your-username/Breast-Cancer-Classification.git
cd Breast-Cancer-Classification
pip install -r requirements.txt
```

---

##  How to Run

###  Run via Jupyter Notebook

```bash
jupyter notebook Breast_Cancer_Project.ipynb
```

###  Run on Google Colab

Upload the `.ipynb` file and run all cells.

---

##  Results

| Model             | Accuracy   | Recall     | F1 Score   | ROC-AUC    |
| ----------------- | ---------- | ---------- | ---------- | ---------- |
| **Optimized SVM** | **0.9824** | **0.9861** | **0.9861** | **0.9937** |
| ANN               | 0.9649     | 0.9583     | 0.9718     | 0.9940     |
| Random Forest     | 0.9561     | 0.9722     | 0.9655     | 0.9937     |
| KNN               | 0.9561     | 0.9722     | 0.9655     | 0.9788     |
| Naive Bayes       | 0.9386     | 0.9583     | 0.9517     | 0.9877     |
| Decision Tree     | 0.9123     | 0.9027     | 0.9285     | 0.9156     |

---

##  Best Model — Optimized SVM

###  Hyperparameters (via GridSearchCV)

```
C = 0.1
kernel = "linear"
gamma = "scale"
degree = 2
```

###  Final Performance

* **98% Accuracy**
* **98.6% Recall**
* **0.99 ROC-AUC (highest)**

This makes the optimized SVM highly reliable for medical diagnostic tasks where **minimizing false negatives** is crucial.

---

##  License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

##  Author

**Muhammad Hanzala Khan**
AI Student & Machine Learning Enthusiast
