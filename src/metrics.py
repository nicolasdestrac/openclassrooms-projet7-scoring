import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    make_scorer,
)

def confusion_at_threshold(y_true, y_prob, threshold: float):
    """Matrice de confusion + ratios au seuil donné."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    recall = tp / (tp + fn) if (tp + fn) else np.nan        # TPR
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": float(recall),
        "precision": float(precision),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }

def business_cost(y_true, y_prob, threshold, cost_fn: float, cost_fp: float) -> float:
    """Coût = FN*cost_fn + FP*cost_fp au seuil donné."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return float(fn * cost_fn + fp * cost_fp)

def _max_constant_cost(y_true, cost_fn: float, cost_fp: float) -> float:
    """
    Coût de la pire politique 'constante' :
      - tout refuser  -> FN = #positifs -> coût = pos * cost_fn
      - tout accepter -> FP = #négatifs -> coût = neg * cost_fp
    On normalise avec le max des deux (borne supérieure raisonnable).
    """
    y_true = np.asarray(y_true, dtype=int)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    return float(max(pos * cost_fn, neg * cost_fp))

def business_score_from_cost(cost: float, y_true, cost_fn: float, cost_fp: float) -> float:
    """
    Score métier normalisé dans [0,1] :
      score = 1 - (cost / max_constant_cost)
    - 1.0  = parfait (coût nul)
    - 0.0  = aussi mauvais que la pire politique constante
    - on clip dans [0,1] (si jamais cost > max_constant_cost)
    """
    max_c = _max_constant_cost(y_true, cost_fn, cost_fp)
    if max_c <= 0:
        # cas dégénéré (ex: pas de positifs et pas de négatifs)
        return 1.0 if cost <= 0 else 0.0
    score = 1.0 - (float(cost) / max_c)
    return float(np.clip(score, 0.0, 1.0))

def best_cost_and_threshold(y_true, y_prob, cost_fn: float, cost_fp: float, grid_points: int = 501):
    """Balaye [0,1] et renvoie (coût minimal, seuil optimal)."""
    grid = np.linspace(0.0, 1.0, int(grid_points))
    costs = [business_cost(y_true, y_prob, t, cost_fn, cost_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(costs[i]), float(grid[i])

def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def evaluate_all(y_true, y_prob, cost_fn: float, cost_fp: float, grid_points: int = 501):
    """
    Renvoie un paquet de métriques globales (AUC, AP, Brier, KS),
    le seuil métier optimal, le coût au seuil optimal (et ramené /10k),
    la matrice de confusion (au seuil optimal et à 0.5),
    et le SCORE MÉTIER NORMALISÉ dans [0,1].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Globales
    auc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))

    # Seuil métier optimal + coût
    best_cost, best_thr = best_cost_and_threshold(
        y_true, y_prob, cost_fn=cost_fn, cost_fp=cost_fp, grid_points=grid_points
    )
    n = len(y_true)
    cost_per10k = (best_cost / n * 10000.0) if n else np.nan

    # Score métier normalisé [0,1]
    biz_score = business_score_from_cost(best_cost, y_true, cost_fn, cost_fp)

    # Confusions
    cm_opt = confusion_at_threshold(y_true, y_prob, best_thr)
    cm_05 = confusion_at_threshold(y_true, y_prob, 0.5)

    return {
        "auc": auc,
        "ap": ap,
        "brier": brier,
        "ks": ks,
        "business_cost": float(best_cost),
        "business_cost_per10k": float(cost_per10k) if not np.isnan(cost_per10k) else np.nan,
        "threshold_opt": float(best_thr),
        "business_score": float(biz_score),
        "confusion_opt": cm_opt,
        "confusion_05": cm_05,
    }

# --- Scorer sklearn basé sur le BUSINESS SCORE (dans [0,1], plus haut = mieux) ---
def _scores_from_estimator(estimator, X):
    """Retourne un score continu dans [0,1] pour chaque ligne (proba préférée)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        # secours: on rabat en [0,1]
        return 1.0 / (1.0 + np.exp(-np.asarray(s, dtype=float)))
    # dernier recours
    return estimator.predict(X)

def make_business_scorer(fn_cost=10.0, fp_cost=1.0, grid=501):
    """
    Scorer à passer à GridSearchCV/RandomizedSearchCV.
    Retourne le 'business_score' (dans [0,1]) => greater_is_better=True implicite.
    """
    def _scorer(estimator, X, y_true):
        y_true = np.asarray(y_true, dtype=int)
        y_score = _scores_from_estimator(estimator, X)
        res = evaluate_all(y_true, y_score, cost_fn=float(fn_cost), cost_fp=float(fp_cost), grid_points=int(grid))
        return float(res["business_score"])  # on maximise ce score
    return _scorer
