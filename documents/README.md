# Bio-Inspired Feature Selection for Breast Cancer Diagnosis

## 1. Introduction

This project explores the use of bio-inspired optimization methods to select useful features for breast cancer diagnosis. It uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, which contains measurements computed from digitized images of breast masses. Each case is labeled benign or malignant.

## 2. Problem Statement

The dataset includes many measurements, and using every feature can make a model more complex than necessary. The goal is to identify a smaller subset of features that can distinguish benign from malignant cases while retaining strong classification performance.

## 3. Approach

Four search algorithms are compared: Simulated Annealing (SA), Genetic Algorithm (GA), Binary Particle Swarm Optimization (Binary PSO), and the Firefly Algorithm. Each algorithm searches for a feature subset using a fitness score that combines cross-validated F1 performance with a preference for selecting fewer features. A Random Forest using all features serves as the baseline. The selected subsets are then evaluated on a held-out test set.

## 4. Outputs

Running the project produces a comparison of the baseline and optimized feature subsets, including accuracy, precision, recall, F1, ROC AUC, number of selected features, and optimization time. Results are saved in `results/`, and metric comparison and convergence charts are saved in `plots/`.
