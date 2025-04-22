**Classification of Sports Injuries through Biomechanical Data Analysis and Machine Learning Models**

A full workflow for predicting running‑related injuries from 3D gait biomechanics using RF, SVM, MLP, KNN & Naive Bayes. This project implements a full ML workflow—data preprocessing, feature engineering, model training and evaluation—for binary classification of running‑related injuries. We compare Random Forest, SVM, MLP, KNN and Naive Bayes on a dataset of kinematic gait variables collected via 3D motion capture.

**Background**

This repository implements a pipeline for binary classification of running‑related injuries. We leverage kinematic gait variables collected with multi‑camera 3D motion capture and compare five supervised learning algorithms to identify patterns indicative of injury risk.

**Data**

**Source**: Publicly available gait biomechanics dataset  
**Samples**: 1,798 individuals (healthy + injured)  
**Features**: 22 kinematic & demographic variables (after filtering)  
**Target**: `InjDefn_binary` (0 = no‐injury/nan, 1 = injury)
