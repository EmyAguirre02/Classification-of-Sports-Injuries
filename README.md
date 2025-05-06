# Sports Injuries Classification Using Machine Learning Models on Biomechanical Data

## Overview

This repository contains code and documentation for the classification of running-related injuries using biomechanical kinematic data from the Running Injury Clinic. We implement and compare five classic supervised learning models to predict injury type based on movement patterns.

## Table of Contents

* [Project Structure](#project-structure)
* [Dataset](#dataset)
* [Environment Setup](#environment-setup)
* [Usage](#usage)
* [Model Architectures](#model-architectures)
* [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
* [Directory Layout](#directory-layout)
* [Contact](#contact)
  
## Project Structure
```
├── src/                       # Source code for model training and evaluation
├── Data/                      # Project data
│   ├── Raw Dataset/           # Raw biomechanical kinematic dataset
├── Notebook/                  # Jupyter notebook for exploration & visualization
├── Docs/                      # Documentation & manuscripts
│   ├── Complementary Paper/   # Supplementary manuscript 
│   ├── Report/                # Technical report
│   └── Presentation/          # Slide decks and presentations
├── Assets/                    # Supporting assets
│   └── Fonts/                 # Custom fonts for graphics
├── Requirements.txt           # Python dependencies
├── .gitattributes             # Git attributes configuration
└── README.md                  # Project overview and setup instructions
```

## Dataset

We use the **Running Injury Clinic Kinematic Dataset**, which contains biomechanical time-series and summary features collected from runners classified into various injury categories (e.g., patellofemoral pain, Achilles tendinopathy). Key characteristics:

* **Features**: Joint angles, moments, and temporal gait parameters
* **Labels**: Injury type (binary or multi-class)
* **Splits**: 70% training (with 10‑fold CV), 30% hold-out test

## Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/running-injury-classification.git
   cd running-injury-classification
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Preprocess data**:

   ```bash
   python src/preprocess.py --input data/raw --output data/processed
   ```
2. **Train models**:

   ```bash
   python src/train_models.py --data data/processed --models all
   ```
3. **Evaluate performance**:

   ```bash
   python src/evaluate.py --models results/models --metrics accuracy precision recall f1 auc
   ```
4. **Visualize results**:
   Open the Jupyter notebooks in `notebooks/` for exploratory analysis and plotting.

## Model Architectures

We compare five supervised classifiers:

* **Random Forest (RF)**: Ensemble of decision trees, default `n_estimators=100`, criterion=`entropy`.
* **Support Vector Machine (SVM)**: Linear, polynomial (degree=3), and RBF kernels.
* **K-Nearest Neighbors (KNN)**: k=3, 5, 7 with Euclidean distance.
* **Gaussian Naive Bayes (GNB)**: Assumes feature independence and Gaussian distributions.
* **Multi-Layer Perceptron (MLP/FFBP)**:

  * **MLP1 (FFBP1)**: Two hidden layers (2×input\_dim, input\_dim), ReLU activations; single-unit output layer with sigmoid activation; trained using the Adam optimizer (lr=1e-4) and binary crossentropy loss.
  * **MLP2 (FFBP2)**: Two hidden layers (½×input\_dim, input\_dim), ReLU activations; single-unit sigmoid output; includes Dropout (0.5) for regularization; trained with Adam (lr=1e-4) and binary crossentropy.
  * **MLP3 (FFBP3)**: Single hidden layer (input\_dim), ReLU activation; single-unit sigmoid output; trained with Adam (lr=1e-4) using binary crossentropy loss.

## Evaluation Metrics

We assess model performance using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score** (primary metric for imbalanced classes)
* **AUC-ROC**
* **Confusion Matrix**

Cross-validation (10‑fold) is used on the training split; final metrics are reported on the hold-out test set.

## Results

Detailed evaluation reports and plots are available in the `results/` directory. Highlights:

* **FFBP2** achieved the highest average F1-Score (0.XX ± 0.XX) across folds.
* Random Forest showed robust precision but lower recall in certain injury classes.
* SVM with RBF kernel balanced precision and recall moderately well.

## Directory Layout

Refer to the [Project Structure](#project-structure) section for an overview of folders and files.
\

## Contact

María Emilia Aguirre 





