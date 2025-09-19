import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

def business_cost(y_true, y_prob, threshold, cost_fn: float, cost_fp: float):
    y_pred = (y_prob >= threshold).astype(int)
    fp = ((y_pred==1) & (y_true==0)).sum()
    fn = ((y_pred==0) & (y_true==1)).sum()
    return float(fn*cost_fn + fp*cost_fp)

def best_cost_and_threshold(y_true, y_prob, cost_fn: float, cost_fp: float, grid_points: int = 501):
    grid = np.linspace(0.0, 1.0, grid_points)
    costs = [business_cost(y_true, y_prob, t, cost_fn, cost_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(costs[i]), float(grid[i])

def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def evaluate_all(y_true, y_prob, cost_fn: float, cost_fp: float, grid_points: int = 501):
    c, t = best_cost_and_threshold(y_true, y_prob, cost_fn, cost_fp, grid_points)
    return {
        "auc":   float(roc_auc_score(y_true, y_prob)),
        "ap":    float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ks":    ks_stat(y_true, y_prob),
        "business_cost": c,
        "best_threshold": t,
        "business_cost_per10k": c/len(y_true)*10000.0
    }
