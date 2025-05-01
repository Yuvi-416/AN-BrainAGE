# AN-BrainAGE
**Does Restrictive Anorexia Nervosa Impact Brain Aging? A Machine Learning Approach to Estimate Age Based on Brain Structure**

This repository contains code and pretrained models for the brain age prediction pipeline used in our study. We used machine learning (Support Vector Regression, Deep Kernel Learning, Gaussian Process Regression [DKL-GPR], and BrainAgeR) to estimate brain age from structural MRI features and analyze brain-predicted age difference (brain-PAD) in participants with anorexia nervosa (AN) and healthy controls (HC).

## 🧠 Overview

**Key files and folders:**
- `train_test_DKL_GPR.py`: Script to train the DKL-GPR model
- `train_test_SVR.py`: Script to train the SVR model
- `Model_DKL_GPR_1000.pt`: Pretrained DKL-GPR model

**Models available:**
- DKL-GPR (used in final manuscript analyses)

## 📄 Features Used
We used **377 features** per participant:
- **Cortical features**: 68 Desikan-Killiany regions × 5 measures (volume, surface area, mean curvature, mean thickness, white matter volume) → 340 features.
- **Subcortical features**: Volumes of 37 subcortical regions (excluding brainstem).

All features were extracted using **FreeSurfer v7.3.2**.

## 📂 Preparing CSV Files
Your CSV input should have the following columns:
```plaintext
Filenames, feature1, feature2, ..., feature377, target
