# MALLORN Astronomical Classification – TDE Detection

This repository contains the implementation and analysis for a Master’s-level machine learning project on **photometric classification of tidal disruption events (TDEs)** using **traditional (non–deep learning) methods**.

The work is based on the **MALLORN Astronomical Classification Challenge** dataset and focuses on feature engineering from sparse light curves, ensemble learning, and statistically validated evaluation.

---

## Project Overview

Modern time-domain surveys (e.g. LSST) detect large numbers of transient astronomical events, while spectroscopic follow-up remains limited.  
This motivates the need for **photometric pre-classification** to prioritize scientifically interesting candidates.

**Learning task:**  
Binary classification of transient events into **TDE vs. non-TDE** using engineered light-curve features.

**Key challenges:**
- Strong class imbalance (~5% TDEs)
- Sparse and irregular light-curve sampling
- Significant feature overlap between transient classes

To address these challenges, a **stacked ensemble learning approach** with heterogeneous base models is adopted and statistically validated.

---

## Repository Structure

.
├── data.ipynb # Main notebook (EDA, modeling, evaluation)
├── extract_data.py # Feature extraction from light curves
├── plot_lightcurve.py # Light-curve visualization
├── StackingEnsemble.py # Custom stacking ensemble implementation
├── custom_wrapped_NN.py # Shallow neural network wrapper
├── classify.py # Model training and evaluation
├── create_submission.ipynb # Kaggle submission pipeline
├── requirements.txt # Python dependencies
├── meta_data/ # Saved models and OOF predictions
└── submission_files/ # Generated submission CSVs

---


## Method Summary

- **Feature engineering:** per-filter statistical and temporal descriptors (~62 features)
- **Base models:** Logistic Regression, Gaussian Naive Bayes, QDA, SVMs, Random Forest, XGBoost, shallow neural network
- **Ensemble:** stacked model with Decision Tree meta-learner trained on out-of-fold predictions
- **Evaluation:** F1 score, ROC–AUC, PCA/t-SNE projections, OOF correlation analysis, Friedman and Nemenyi tests

---

## Key Results

- Feature pruning improved F1 score from ~0.22 → ~0.30
- ROC–AUC ≈ 0.84, indicating meaningful discriminative ability
- Statistical tests confirm significant heterogeneity among base models
- OOF correlations show complementary error patterns, supporting stacking

Overall performance is primarily constrained by **overlapping feature distributions**, rather than model capacity.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```


### 2. Run the main analysis
```bash
data.ipynb
```

---

The notebook contains:
- Exploratory data analysis
- Feature engineering
- Model training
- Statistical evaluation
- Final analysis and discussion

---

## Reproducibility
- Fixed random seeds
- Stratified cross-validation
- Out-of-fold predictions stored for inspection
- Models saved using `joblib`

---

## Academic Context
This repository accompanies a **3–5 page IEEE-style report** submitted for:

**Machine Learning (DV2638)**  
M.Sc. in AI & Machine Learning  
Blekinge Institute of Technology

All design choices are motivated, evaluated, and statistically validated in the report and notebook.

---

## Authors
**Rasmus Eliasson**  
M.Sc. AI & Machine Learning, BTH

**Oskar Flodin**  
M.Sc. AI & Machine Learning, BTH
