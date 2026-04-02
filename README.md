# Day 32 — Decision Trees & Random Forest (AM + PM Combined)

**Week 6 | IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**

---

## Assignment Overview

| Field          | Detail                                                                                                                                                         |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Topics         | Decision Trees, Random Forest, Extra Trees, Bias-Variance Tradeoff, Hyperparameter Tuning, Feature Importance, Cost-sensitive Learning, Model Interpretability |
| Estimated Time | 2–3 hours                                                                                                                                                      |
| Submission     | GitHub commit + Jupyter notebooks, link in Slack `#daily-standup`                                                                                              |
| Due            | Next day 09:15 AM                                                                                                                                              |

---

## Project Structure

```
Day32_DT_RF_Combined/
│
├── README.md
│
├── AM_SESSION/
│   ├── D32_AM_DT_RandomForest.ipynb
│   ├── D32_AM_PartB_ExtraTrees.ipynb
│   └── extra_trees_comparison.md
│
├── PM_SESSION/
│   ├── D32_PM_DT_RF_CaseStudy.ipynb
│   └── gradient_boosting_preview.md
```

---

# AM Session — Foundations & Core Concepts

## Part A — Concept Application (40%)

* Loan dataset (2000 records, 6 features)
* Decision Tree (`max_depth=4`)
* Rule extraction (IF-THEN rules)
* Random Forest tuning (`RandomizedSearchCV`, ROC-AUC)
* Model comparison (Accuracy, F1, ROC-AUC)
* Feature importance (MDI vs Permutation)

### Outputs

* decision_tree_plot.png
* model_comparison.png
* feature_importance_comparison.png

---

## Part B — Extra Trees vs Random Forest (30%)

* Threshold selection comparison
* Speed benchmarking (`n_estimators`)
* Cross-validation comparison
* Feature importance comparison

### Outputs

* threshold_comparison.png
* speed_comparison.png
* cv_comparison.png

---

## Part C — Interview Ready (20%)

* Bias-Variance tradeoff
* Overfitting curve (`max_depth`)
* RF debugging (train=test case)

### Outputs

* bias_variance_diagram.png
* overfitting_curve.png

---

## Part D — AI-Augmented Task (10%)

* Model infographic (DT vs RF vs Logistic Regression)
* Non-technical explanations

---

## AM Model Summary

| Model         | Accuracy | F1    | ROC-AUC | Interpretability |
| ------------- | -------- | ----- | ------- | ---------------- |
| Decision Tree | ~0.91    | ~0.91 | ~0.94   | ★★★★★            |
| Random Forest | ~0.93    | ~0.93 | ~0.97   | ★★☆☆☆            |
| Extra Trees   | ~0.93    | ~0.92 | ~0.97   | ★★☆☆☆            |

---

# PM Session — Applied Case Study

## Part A — Fraud Detection (40%)

* Insurance dataset (3000 records, 8 features)
* Decision Tree (`max_depth=5`, class_weight='balanced')
* Rule extraction
* Random Forest tuned for **recall**
* Full metrics + confusion matrix
* Cost-sensitive evaluation (FN = 10× FP)
* Threshold optimization

### Outputs

* fraud_dt_plot.png
* confusion_matrices.png
* cost_threshold_curve.png
* fraud_feature_importance.png

---

## Part B — Gradient Boosting (30%)

* Boosting vs Bagging
* Residual learning concept
* Comparison table
* Resources + self-check

---

## Part C — Interview Ready (20%)

* 100 vs 1000 trees tradeoff
* `compare_models()` reusable function
* RF variance debugging

### Outputs

* accuracy_saturation.png
* model_comparison_cv.png

---

## Part D — AI-Augmented Task (10%)

* OOB error explanation
* Mathematical validation (≈36.8%)
* When OOB fails
* Code demo

---

## Business Context

| Metric           | Priority  | Reason                     |
| ---------------- | --------- | -------------------------- |
| Recall           | Highest   | Missing fraud is costly    |
| ROC-AUC          | High      | Threshold tuning           |
| Interpretability | Required  | Compliance                 |
| Precision        | Secondary | Manageable false positives |

---

## PM Model Summary

| Model         | Recall | ROC-AUC | Business Cost | Interpretability |
| ------------- | ------ | ------- | ------------- | ---------------- |
| Decision Tree | ~0.80  | ~0.86   | Higher        | ★★★★★            |
| Random Forest | ~0.87  | ~0.92   | Lower         | ★★☆☆☆            |

---

## How to Run

```bash
pip install scikit-learn pandas numpy matplotlib scipy

jupyter notebook AM_SESSION/D32_AM_DT_RandomForest.ipynb
jupyter notebook AM_SESSION/D32_AM_PartB_ExtraTrees.ipynb
jupyter notebook PM_SESSION/D32_PM_DT_RF_CaseStudy.ipynb
```

---

## Final Takeaways

* Decision Trees → interpretable but overfit
* Random Forest → best overall performance
* Extra Trees → faster, more random
* Gradient Boosting → reduces bias (advanced)

---

## Recommended Deployment

* Use **Random Forest** for prediction
* Use **Decision Tree rules** for explanation
* Apply **threshold tuning** for cost-sensitive problems

---
