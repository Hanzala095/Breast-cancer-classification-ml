```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

#  Breast Cancer Prediction Using Machine Learning  
A complete end-to-end Machine Learning project for predicting whether a breast tumor is **benign** or **malignant**, using multiple ML models, preprocessing, EDA, evaluation metrics, hyperparameter tuning, and final optimized SVM classifier.

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
This project implements a **complete ML workflow** for breast cancer classification:

- Data Loading & Cleaning  
- Exploratory Data Analysis (EDA)  
- Train–Test split  
- Feature Scaling  
- Training **six ML models**  
- Full evaluation using Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Hyperparameter tuning using GridSearchCV  
- Final optimized **Support Vector Machine (SVM)** model  

---

##  Dataset
**Breast Cancer Wisconsin Diagnostic Dataset**  
Included through `sklearn.datasets.load_breast_cancer()`.

- 569 samples  
- 30 numerical features  
- Binary target:
  - `0 = Malignant`
  - `1 = Benign`

No external download required.

---

##  Features Implemented
 Full preprocessing pipeline  
 Exploratory data analysis  
 Model training:  
   - Decision Tree  
   - Random Forest  
   - Support Vector Machine  
   - KNN  
   - Naive Bayes  
   - Artificial Neural Network  
 Model comparison table  
 ROC-AUC analysis  
 Hyperparameter tuning  
 Final optimized SVM model  

---

##  Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Google Colab / Jupyter Notebook  

---

##  Repository Structure

```

 Breast-Cancer-Classification
│
├── CCP_Report.pdf             # Full written report
├── Breast_Cancer_Project.ipynb # Complete code implementation
├── README.md                  # Documentation
├── LICENSE                    # MIT License
└── requirements.txt           # Python dependencies

````

---

##  Installation

```bash
git clone https://github.com/your-username/Breast-Cancer-Classification.git
cd Breast-Cancer-Classification
pip install -r requirements.txt
````

---

##  How to Run the Project

### **1. Run via Jupyter Notebook**

```bash
jupyter notebook Breast_Cancer_Project.ipynb
```

### **2. Run on Google Colab**

Upload the `.ipynb` file directly and run all cells.

---

##  Results Summary

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

Hyperparameters selected using GridSearchCV:

```
C = 0.1  
kernel = 'linear'  
gamma = 'scale'  
degree = 2  
```

The optimized SVM achieved:

*  **98% accuracy**
*  **Excellent recall (98.6%)**
*  **Highest ROC-AUC among all models**

This makes it highly reliable for medical diagnostic tasks where minimizing false negatives is essential.

---

##  License

This project is licensed under the **MIT License**.
See `LICENSE` for details.

---

##  Author

**Muhammad Hanzala Khan**
AI Student & Machine Learning Enthusiast


```



