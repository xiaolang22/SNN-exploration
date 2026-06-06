from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def extract_v100(spike_counts: np.ndarray) -> np.ndarray:
    active = np.where(spike_counts > 0)[0]
    if len(active) == 0:
        return np.zeros(100, dtype=np.int32)
    sorted_idx = active[np.argsort(spike_counts[active])[::-1]]
    top100 = sorted_idx[:100]
    result = np.zeros(100, dtype=np.int32)
    result[: len(top100)] = top100 + 1
    return result


def classify_round(features: np.ndarray, labels: np.ndarray,
                   n_folds: int = 10) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return 0.0, 0.0
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in kf.split(features):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(features[train_idx], labels[train_idx])
        accs.append(clf.score(features[test_idx], labels[test_idx]))
    return float(np.mean(accs)), float(np.std(accs))


def compute_confusion_matrix(features: np.ndarray, labels: np.ndarray,
                             n_folds: int = 10) -> np.ndarray:
    unique_labels = np.unique(labels)
    preds = np.empty_like(labels)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(features):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(features[train_idx], labels[train_idx])
        preds[test_idx] = clf.predict(features[test_idx])
    return confusion_matrix(labels, preds, labels=unique_labels)
