
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from src.fitness import FeatureSelectionFitness
from src.optimizers import (
    simulated_annealing, genetic_algorithm,
    binary_pso, firefly_algorithm
)

DATA_PATH = Path("data/wdbc.data")
RESULTS = Path("results")
PLOTS = Path("plots")

RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

names = [
    "radius", "texture", "perimeter", "area",
    "smoothness", "compactness", "concavity",
    "concave_points", "symmetry", "fractal_dimension"
]
columns = ["ID", "Diagnosis"] + [
    f"{name}{suffix}"
    for suffix in [1, 2, 3]
    for name in names
]

df = pd.read_csv(DATA_PATH, header=None, names=columns)
X = df.drop(columns=["ID", "Diagnosis"])
y = df["Diagnosis"].map({"B": 0, "M": 1})

if y.isna().any():
    raise ValueError("Unexpected diagnosis labels in dataset.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

def evaluate_test(mask, label, elapsed=0.0):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train.loc[:, mask], y_train)

    pred = model.predict(X_test.loc[:, mask])
    prob = model.predict_proba(X_test.loc[:, mask])[:, 1]

    return {
        "Method": label,
        "Features": int(np.asarray(mask).sum()),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, prob),
        "Optimization_Seconds": elapsed
    }

all_features = np.ones(X.shape[1], dtype=bool)
rows = [evaluate_test(all_features, "Baseline RF")]

fitness = FeatureSelectionFitness(X_train, y_train)
n = X_train.shape[1]

algorithms = [
    ("SA", simulated_annealing),
    ("GA", genetic_algorithm),
    ("Binary PSO", binary_pso),
    ("Firefly", firefly_algorithm)
]

histories = {}

for name, algorithm in algorithms:
    print(f"\nRunning {name}...")
    start = time.perf_counter()

    mask, history = algorithm(
        fitness, n, seed=42
    )

    elapsed = time.perf_counter() - start
    histories[name] = history

    print("Selected features:", int(mask.sum()))
    print("Feature names:", list(X.columns[mask]))

    rows.append(evaluate_test(mask, name, elapsed))

    pd.Series(
        X.columns[mask], name="Selected_Feature"
    ).to_csv(RESULTS / f"{name.lower().replace(' ', '_')}_features.csv",
             index=False)

results = pd.DataFrame(rows)
results.to_csv(RESULTS / "comparison.csv", index=False)

print("\n===== FINAL TEST COMPARISON =====")
print(results.round(4).to_string(index=False))

# Comparison plots
for metric in ["Accuracy", "F1", "ROC_AUC"]:
    plt.figure()
    plt.bar(results["Method"], results[metric])
    plt.ylabel(metric)
    plt.title(f"Model Comparison: {metric}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS / f"{metric.lower()}_comparison.png", dpi=150)
    plt.close()

plt.figure()
for name, history in histories.items():
    plt.plot(range(1, len(history) + 1), history, label=name)

plt.xlabel("Iteration / Generation")
plt.ylabel("Best fitness so far")
plt.title("Optimization Convergence")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "convergence.png", dpi=150)
plt.close()

print("\nSaved results to:", RESULTS.resolve())
print("Saved plots to:", PLOTS.resolve())
