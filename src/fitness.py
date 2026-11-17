
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


class FeatureSelectionFitness:
    def __init__(self, X, y, random_state=42):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.n_features = self.X.shape[1]
        self.cache = {}

        self.cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=random_state
        )

    def evaluate(self, mask):
        mask = np.asarray(mask, dtype=bool)

        # A solution must select at least one feature.
        if not mask.any():
            return 0.0

        key = tuple(mask.tolist())

        if key in self.cache:
            return self.cache[key]

        X_selected = self.X[:, mask]

        model = RandomForestClassifier(
            n_estimators=60,
            random_state=42,
            n_jobs=-1
        )

        scores = cross_val_score(
            model,
            X_selected,
            self.y,
            cv=self.cv,
            scoring="f1",
            n_jobs=1
        )

        mean_f1 = scores.mean()
        feature_ratio = mask.sum() / self.n_features

        # Reward predictive performance and fewer features.
        fitness = (
            0.9 * mean_f1
            + 0.1 * (1 - feature_ratio)
        )

        self.cache[key] = float(fitness)
        return float(fitness)
