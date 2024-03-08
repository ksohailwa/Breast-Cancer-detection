# Breast Cancer Detection using Logistic Regression

## Overview
This repository contains code for building a machine learning model to predict breast cancer using logistic regression. The dataset used for training and testing is the `breast_cancer.csv` dataset.

## Dataset
The dataset contains various features extracted from breast cancer biopsies, along with the corresponding diagnosis (M = malignant, B = benign).

## Files
1. `breast_cancer.csv`: Dataset file containing biopsy features and diagnosis labels.
2. `logistic_regression_breast_cancer.ipynb`: Jupyter Notebook containing the code for training, testing, and evaluating the logistic regression model.
3. `README.md`: This file, providing an overview of the project and instructions for running the code.

## Instructions
To run the code, follow these steps:

1. Ensure you have Python and necessary libraries installed (Pandas, NumPy, Scikit-learn).
2. Clone the repository to your local machine or download the files.
3. Open the Jupyter Notebook `logistic_regression_breast_cancer.ipynb` in an environment such as Google Colab or Jupyter Notebook.
4. Execute the code cells in the notebook sequentially to train the logistic regression model, make predictions, and evaluate its performance.
5. Ensure that the dataset file `breast_cancer.csv` is in the same directory as the notebook or provide the correct path to the dataset.

## Results
The logistic regression model achieves an accuracy of 96.70% on the test set with a standard deviation of 1.97% when evaluated using k-fold cross-validation.

## References
- Dataset Source: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

For any questions or issues, feel free to contact the repository owner.
