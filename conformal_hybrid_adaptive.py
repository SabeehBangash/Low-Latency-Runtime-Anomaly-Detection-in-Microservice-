"""
Conformal-calibrated hybrid anomaly detector  (v2, reviewed)
------------------------------------------------------------
Compares, over R repeated random splits (default 30):
  A  AE + 95th-percentile of calibration normals      (your CURRENT rule)
  B  AE-only          + split-conformal threshold
  C  XGBoost-only     + split-conformal threshold
  D  Hybrid (AE+XGB)  + split-conformal threshold      (NEW score)
  E  Hybrid           + adaptive (ACI-style) threshold (NEW update rule)

Run:  python conformal_hybrid_adaptive.py --csv path/to/dataset.csv \
        --label-col label --normal-labels normal inter_service_normal
Add   --drift 0.5   for the synthetic drift stress test (clearly label it
                    "synthetic drift" in the paper).
"""
import argparse, math, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
    def make_xgb(seed): return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, random_state=seed, eval_metric="logloss", n_jobs=1)
except ImportError:                       # sandbox fallback only
    from sklearn.ensemble import GradientBoostingClassifier
    def make_xgb(seed): return GradientBoostingClassifier(random_state=seed)

try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(1)
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    class AEnet(nn.Module):                      # same architecture as train_models.py
        def __init__(self, d):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d, 24), nn.ReLU(), nn.Linear(24, 12), nn.ReLU(),
                                         nn.Linear(12, 6), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 24), nn.ReLU(),
                                         nn.Linear(24, d))
        def forward(self, x): return self.decoder(self.encoder(x))

def fit_ae_torch(Xn, seed):
    """Adam lr 1e-3, batch 16, 80 epochs, MSE -- same as train_models.py. Returns error function."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = AEnet(Xn.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=0.001); crit = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(Xn, dtype=torch.float32)), batch_size=16, shuffle=True, generator=g)
    m.train()
    for _ in range(80):
        for (b,) in loader:
            opt.zero_grad(); loss = crit(m(b), b); loss.backward(); opt.step()
    m.eval()
    def err(Z):
        with torch.no_grad():
            R = m(torch.tensor(Z, dtype=torch.float32)).numpy()
        return ((Z - R) ** 2).mean(1)
    return err

def fit_ae_mlp(Xn, seed):                         # sandbox fallback only
    ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=3000, random_state=seed).fit(Xn, Xn)
    return lambda Z: ((ae.predict(Z) - Z) ** 2).mean(1)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--label-col", default="label")
ap.add_argument("--normal-labels", nargs="+", default=["normal"])
ap.add_argument("--repeats", type=int, default=30)
ap.add_argument("--w", type=float, default=0.5, help="hybrid weight on AE (fixed in advance)")
ap.add_argument("--gamma", type=float, default=0.05)
ap.add_argument("--window", type=int, default=200)
ap.add_argument("--drift", type=float, default=0.0, help="synthetic drift (std units at stream end)")
ap.add_argument("--out", default="results")
ap.add_argument("--seed0", type=int, default=1000, help="first seed; use 0 to match train_models_multiseed.py seeds 0..N-1")
ap.add_argument("--split", choices=["602020", "table5"], default="602020",
                help="602020 = 60/20/20 stratified by class (default); table5 = same protocol as train_models.py (70/30, then 80/20)")
ap.add_argument("--ae", choices=["torch", "mlp"], default="torch", help="torch = your Autoencoder (use this); mlp = test stand-in")
ap.add_argument("--feature-json", default=None, help="feature_columns.json used by train_models.py (list of names)")
ap.add_argument("--exclude", nargs="*", default=[], help="extra non-feature columns to drop")
a = ap.parse_args()

df = pd.read_csv(a.csv)
y_lab = df[a.label_col].astype(str).values
y = (~np.isin(y_lab, a.normal_labels)).astype(int)          # 1 = anomaly
import json
if a.feature_json:
    fj = json.load(open(a.feature_json))
    if isinstance(fj, dict):
        fj = next(v for v in fj.values() if isinstance(v, list))
    feats = list(fj)
else:
    feats = [c for c in df.select_dtypes("number").columns if c != a.label_col and c not in a.exclude]
X = df[feats].values.astype(float)
print("Non-numeric / dropped columns:", [c for c in df.columns if c not in feats])
print("Feature columns used (%d):" % len(feats), feats)
# leakage guard: a single column that almost perfectly separates normal vs anomaly is suspicious
from sklearn.metrics import roc_auc_score
for j, c in enumerate(feats):
    if len(np.unique(X[:, j])) > 1:
        auc = roc_auc_score(y, X[:, j])
        if max(auc, 1 - auc) > 0.99:
            print(f"  WARNING: column '{c}' has AUC {max(auc,1-auc):.3f} for normal-vs-anomaly -> possible label leakage")
print(f"autoencoder={a.ae} split={a.split}")
print(f"rows={len(df)} features={X.shape[1]} normal={int((y==0).sum())} anomaly={int(y.sum())}")

def conf_thr(cal, alpha):
    """split-conformal: k-th smallest of n cal scores, k=ceil((n+1)(1-alpha)); clamp to n."""
    n = len(cal); k = min(math.ceil((n + 1) * (1 - alpha)), n)
    return np.sort(cal)[k - 1], (math.ceil((n + 1) * (1 - alpha)) > n)

def metrics(flag, ytrue):
    nrm, ano = ytrue == 0, ytrue == 1
    fpr = flag[nrm].mean() if nrm.any() else np.nan
    rec = flag[ano].mean() if ano.any() else np.nan
    tp = flag[ano].sum(); fp = flag[nrm].sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return fpr, rec, f1

def aci_stream(s, ytrue, cal, alpha, gamma, window):
    pool = list(cal); at = alpha; floor = 1 / (len(cal) + 1)
    flags = np.zeros(len(s), int)
    for i, (si, yi) in enumerate(zip(s, ytrue)):
        thr, _ = conf_thr(np.array(pool[-window:]), min(max(at, floor), 0.5))
        flags[i] = int(si > thr)
        if yi == 0:                       # feedback ONLY from confirmed-normal samples
            at += gamma * (alpha - flags[i])
            pool.append(float(si))        # anomalies never enter the calibration pool
    return flags

rows = []
for r in range(a.repeats):
    seed = a.seed0 + r
    idx = np.arange(len(y))
    if a.split == "602020":
        i_tr, i_rest = train_test_split(idx, test_size=0.40, stratify=y_lab, random_state=seed)
        i_cal, i_te = train_test_split(i_rest, test_size=0.50, stratify=y_lab[i_rest], random_state=seed)
    else:                                   # identical to train_models.py: 70/30 on binary label, then 80/20
        i_pool, i_te = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        i_tr, i_cal = train_test_split(i_pool, test_size=0.20, stratify=y[i_pool], random_state=seed)
    sc = StandardScaler().fit(X[i_tr])                       # scaler fit on TRAIN only
    Xtr, Xca, Xte = sc.transform(X[i_tr]), sc.transform(X[i_cal]), sc.transform(X[i_te])
    ytr, yca, yte = y[i_tr], y[i_cal], y[i_te]
    rng = np.random.RandomState(seed); order = rng.permutation(len(i_te))   # stream order
    Xte, yte = Xte[order], yte[order]
    if a.drift > 0:                                           # synthetic drift on test stream
        Xte = Xte + a.drift * np.linspace(0, 1, len(Xte))[:, None]

    if a.ae == "torch" and not HAVE_TORCH:
        raise SystemExit("PyTorch not installed in this environment; cannot use --ae torch")
    err = (fit_ae_torch if a.ae == "torch" else fit_ae_mlp)(Xtr[ytr == 0], seed)
    xgb = make_xgb(seed).fit(Xtr, ytr)
    lg = lambda Z: np.clip(np.log(np.clip(xgb.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6) /
                                 (1 - np.clip(xgb.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6))), -10, 10)
    # score components, standardised with TRAIN-only statistics (fixed functions -> conformal-valid)
    la_tr, lx_tr = np.log(err(Xtr) + 1e-12), lg(Xtr)
    za = lambda Z: (np.log(err(Z) + 1e-12) - la_tr.mean()) / la_tr.std()
    zx = lambda Z: (lg(Z) - lx_tr.mean()) / lx_tr.std()
    S = {"AE": (za(Xca), za(Xte)), "XGB": (zx(Xca), zx(Xte)),
         "Hybrid": (a.w * za(Xca) + (1 - a.w) * zx(Xca), a.w * za(Xte) + (1 - a.w) * zx(Xte))}
    cal_n = yca == 0

    # Row A = your CURRENT rule, exactly: 95th percentile of the RAW reconstruction errors of calibration normals
    thr = np.percentile(err(Xca)[cal_n], 95)
    rows.append(dict(rep=r, method="A: AE + 95th pct", alpha=np.nan, **dict(zip(
        ["FPR", "Recall", "F1"], metrics((err(Xte) > thr).astype(int), yte))), n_cal=cal_n.sum(), clamped=False))
    for alpha in (0.10, 0.05):
        for name, lab in (("AE", "B: AE conformal"), ("XGB", "C: XGB conformal"), ("Hybrid", "D: Hybrid conformal")):
            thr, cl = conf_thr(S[name][0][cal_n], alpha)
            rows.append(dict(rep=r, method=lab, alpha=alpha, **dict(zip(["FPR", "Recall", "F1"],
                        metrics((S[name][1] > thr).astype(int), yte))), n_cal=cal_n.sum(), clamped=cl))
        fl = aci_stream(S["Hybrid"][1], yte, S["Hybrid"][0][cal_n], alpha, a.gamma, a.window)
        fa = aci_stream(S["AE"][1], yte, S["AE"][0][cal_n], alpha, a.gamma, a.window)
        rows.append(dict(rep=r, method="F: AE + adaptive", alpha=alpha, **dict(zip(["FPR", "Recall", "F1"],
                    metrics(fa, yte))), n_cal=cal_n.sum(), clamped=False))
        rows.append(dict(rep=r, method="E: Hybrid + adaptive", alpha=alpha, **dict(zip(["FPR", "Recall", "F1"],
                    metrics(fl, yte))), n_cal=cal_n.sum(), clamped=False))

R = pd.DataFrame(rows)
summ = (R.groupby(["method", "alpha"], dropna=False)[["FPR", "Recall", "F1", "n_cal"]]
          .agg(["mean", "std"]).round(3))
os.makedirs(a.out, exist_ok=True)
tag = (f"_drift{a.drift}" if a.drift else "") + ("" if a.split == "602020" else "_table5split") + ("" if a.seed0 == 1000 else f"_seed0{a.seed0}")
R.to_csv(f"{a.out}/conformal_ablation_per_repeat{tag}.csv", index=False)
summ.to_csv(f"{a.out}/conformal_ablation_summary{tag}.csv")
print(summ.to_string())
print("\nCoverage check: for alpha=0.10/0.05, mean FPR of conformal rows should be <= alpha.")
print("Rows with clamped=True had too few calibration normals for a strict guarantee:",
      int(R.clamped.sum()))
