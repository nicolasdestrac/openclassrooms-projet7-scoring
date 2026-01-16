import numpy as np
from src.metrics import (
    confusion_at_threshold, business_cost, best_cost_and_threshold,
    evaluate_all, business_score_from_cost
)

def test_confusion_and_cost():
    y_true = np.array([0,0,1,1,1,0,1,0])
    y_prob = np.array([0.1,0.2,0.9,0.8,0.7,0.3,0.6,0.4])
    cm = confusion_at_threshold(y_true, y_prob, 0.5)
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == len(y_true)

    c = business_cost(y_true, y_prob, threshold=0.5, cost_fn=10.0, cost_fp=1.0)
    assert c >= 0

def test_best_threshold_and_evaluate_all():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=100)
    y_prob = rng.rand(100)
    best_cost, thr = best_cost_and_threshold(y_true, y_prob, 10.0, 1.0)
    assert 0.0 <= thr <= 1.0
    assert best_cost >= 0

    res = evaluate_all(y_true, y_prob, 10.0, 1.0, grid_points=101)
    for k in ["auc","ap","brier","ks","business_cost","threshold_opt","business_score"]:
        assert k in res

def test_business_score_bounds():
    # score dans [0,1]
    y_true = np.array([0,1,1,0])
    best_cost = 0.0
    s = business_score_from_cost(best_cost, y_true, 10.0, 1.0)
    assert 0.0 <= s <= 1.0
