# Titanic: Binary Classification

Predicting passenger survival on the Titanic using demographic and ticketing data.
Three classifiers compared end-to-end: Logistic Regression, Random Forest, and K-Nearest Neighbours.

## Project Overview

This notebook demonstrates a complete binary classification workflow. The goal is not just
to find the best accuracy number but to compare models deliberately, justify every
preprocessing decision, and evaluate performance beyond accuracy using precision, recall,
F1, and ROC-AUC.

## Setup

```bash
conda create -n titanic-clf python=3.12
conda activate titanic-clf
pip install -r requirements.txt
```

Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/titanic/data)
and place it in `data/`.

## Run

```bash
conda activate titanic-clf
jupyter notebook notebooks/titanic_classification.ipynb
```

## Key Findings

| Model | Accuracy | Recall (survivors) | AUC |
|---|---|---|---|
| Logistic Regression | 80.4% | 66.7% | 0.843 |
| Random Forest | 81.0% | 69.6% | 0.839 |
| KNN (k=7) | 82.1% | 73.9% | 0.849 |

All three models beat the 61.4% dummy baseline. The differences are small. KNN edges
ahead on accuracy and recall. Logistic Regression is the most interpretable: being female
multiplied survival odds by 3.5x, and Pclass was the second strongest predictor.

## Skills Demonstrated

`binary classification` `EDA` `feature engineering` `data preprocessing`
`logistic regression` `random forest` `KNN` `ROC-AUC` `precision-recall`
`scikit-learn` `pandas` `seaborn` `matplotlib`
