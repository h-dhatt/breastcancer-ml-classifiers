
# Machine Learning with a Mathematical Lens — Breast Cancer Dataset

**Author:** You (B.Sc. (Honours) Math & Stats, McMaster)  
**Target Program:** York University — MA in Mathematics & Statistics

## Objectives
1. Apply **Logistic Regression** (probabilistic, convex optimization) and **Decision Tree** (nonparametric, recursive partitioning) to a real biomedical dataset.
2. Compare performance using **accuracy, ROC curves, and cross-validation**.
3. Visualize decision boundaries and feature importances, connecting results back to mathematics of optimization and information gain.

## Key Results
- **Logistic Regression:** Accuracy = 0.982, AUC = 0.995, CV Accuracy ≈ 0.974 ± 0.017
- **Decision Tree:** Accuracy = 0.939, AUC = 0.934, CV Accuracy ≈ 0.923 ± 0.030

> Logistic regression offers a convex optimization framework (guaranteed global optimum), while decision trees provide interpretability via hierarchical splits.

## Deliverables
- **metrics_summary.csv** — accuracy, AUC, cross-validation
- **feature_importances.csv** — decision tree feature ranking
- **fig_roc_curves.png** — ROC comparison
- **fig_decision_boundary.png** — PCA-projected logistic regression boundary
- **fig_feature_importances.png** — top predictors (e.g., mean radius, texture)

## Admissions-Ready Highlights
- Implemented **supervised learning methods** with mathematical rigor: convex optimization (logistic regression) and recursive partitioning (decision trees).
- Evaluated models using **ROC/AUC** and **cross-validation**, ensuring robustness.
- Produced interpretable results (decision boundary visualizations, top predictor ranking).

## How to Run
```bash
pip install numpy pandas matplotlib scikit-learn
python analysis_script.py
```

## Talking Points
- Logistic regression solves a **maximum likelihood problem** with convex loss, ensuring stability.
- Decision trees use **information gain**; feature importances quantify variance reduction.
- ROC/AUC provide a **threshold-independent** measure of classification performance.
