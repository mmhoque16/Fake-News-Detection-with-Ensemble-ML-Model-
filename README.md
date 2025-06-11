This repository implements an explainable machine-learning pipeline for detecting fake news using TF-IDF feature extraction and ensemble classifiers. It includes:

Data ingestion & preprocessing: Merges and cleans a balanced Kaggle dataset of real vs. fake news articles, applies TF-IDF vectorization to capture term importance.

Model training & tuning: Trains Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models; hyperparameters optimized via 5-fold GridSearchCV.

Performance & results: Ensemble methods achieve up to 100 % accuracy on held-out data, with Random Forest and Gradient Boosting selected for interpretability analysis.

Interpretability: Uses Partial Dependence Plots (PDP) and LIME to reveal how features like “reuters” and “21wire” influence predictions, ensuring transparency in classification.

Explore the notebooks and scripts to reproduce the experiments, evaluate model performance, and visualize explanations.
