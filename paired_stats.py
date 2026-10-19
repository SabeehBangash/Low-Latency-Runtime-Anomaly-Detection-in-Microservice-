"""Paired comparison of methods across the repeated splits.
Usage: python paired_stats.py results\\conformal_ablation_per_repeat.csv"""
import sys, pandas as pd, numpy as np
from scipy.stats import wilcoxon
R = pd.read_csv(sys.argv[1])
def get(method, alpha, col):
    d = R[R.method.str.startswith(method)]
    if alpha is not None: d = d[np.isclose(d.alpha, alpha)]
    return d.sort_values("rep")[col].values
pairs = [("D", "B", 0.10, 0.10), ("D", "B", 0.05, 0.05), ("D", "C", 0.10, 0.10), ("D", "C", 0.05, 0.05),
         ("D", "A", 0.10, None), ("E", "D", 0.10, 0.10), ("E", "D", 0.05, 0.05), ("E", "A", 0.05, None),
         ("E", "F", 0.10, 0.10), ("E", "F", 0.05, 0.05), ("F", "B", 0.10, 0.10), ("F", "B", 0.05, 0.05)]
NT = len(pairs) * 3
print(f"{'pair (alpha)':<18}{'metric':<8}{'mean diff':>10}{'95% CI (+/-)':>14}{'wins/ties/losses':>18}{'Wilcoxon p':>12}  Bonferroni(*)")
for x, y, ax, ay in pairs:
    for col in ("Recall", "F1", "FPR"):
        a, b = get(x, ax, col), get(y, ay, col); d = a - b
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        try: p = wilcoxon(a, b).pvalue
        except ValueError: p = float("nan")
        print(f"{x} vs {y} ({ax})".ljust(18) + f"{col:<8}{d.mean():>10.3f}{ci:>14.3f}"
              f"{f'{(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}':>18}{p:>12.4f}  {'*' if p < 0.05 / NT else ''}")
