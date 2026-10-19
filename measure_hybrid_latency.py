"""
Per-sample scoring latency of the hybrid conformal detector (no Docker/Prometheus needed).
Measures, with batch size 1 on CPU, single thread:
  1 Autoencoder error only        2 XGBoost probability only
  3 Hybrid score + static conformal decision
  4 Hybrid score + adaptive threshold update (rolling window of confirmed-normal scores)
Run from the project root:
  python ml\\measure_hybrid_latency.py --csv datasets\\processed\\microservices_anomaly_dataset.csv ^
      --feature-json results\\latency\\feature_columns.json --normal-labels normal inter_service_normal
"""
import argparse, json, math, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(1); HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
try:
    from xgboost import XGBClassifier
    def make_xgb(seed): return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, random_state=seed, eval_metric="logloss", n_jobs=1)
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    def make_xgb(seed): return GradientBoostingClassifier(random_state=seed)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--feature-json", required=True)
ap.add_argument("--label-col", default="label"); ap.add_argument("--normal-labels", nargs="+", default=["normal"])
ap.add_argument("--n", type=int, default=2000, help="timed calls per stage"); ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--ae", choices=["torch", "mlp"], default="torch"); ap.add_argument("--out", default="results")
ap.add_argument("--window", type=int, default=200); ap.add_argument("--alpha", type=float, default=0.10)
a = ap.parse_args(); seed = 0

df = pd.read_csv(a.csv); fj = json.load(open(a.feature_json)); feats = list(fj) if isinstance(fj, list) else next(v for v in fj.values() if isinstance(v, list))
X = df[feats].values.astype(float); lab = df[a.label_col].astype(str).values
y = (~np.isin(lab, a.normal_labels)).astype(int)
idx = np.arange(len(y))
i_tr, i_rest = train_test_split(idx, test_size=0.40, stratify=lab, random_state=seed)
i_cal, i_te = train_test_split(i_rest, test_size=0.50, stratify=lab[i_rest], random_state=seed)
sc = StandardScaler().fit(X[i_tr]); Xtr, Xca, Xte = sc.transform(X[i_tr]), sc.transform(X[i_cal]), sc.transform(X[i_te])
ytr, yca = y[i_tr], y[i_cal]

# ---- models (same as conformal_hybrid_adaptive.py) ----
if a.ae == "torch":
    assert HAVE_TORCH, "PyTorch missing"
    class AEnet(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.encoder = nn.Sequential(nn.Linear(d, 24), nn.ReLU(), nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 6), nn.ReLU())
            s.decoder = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, d))
        def forward(s, x): return s.decoder(s.encoder(x))
    torch.manual_seed(seed); m = AEnet(Xtr.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=0.001); crit = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(Xtr[ytr == 0], dtype=torch.float32)), batch_size=16, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    m.train()
    for _ in range(80):
        for (b,) in dl: opt.zero_grad(); crit(m(b), b).backward(); opt.step()
    m.eval()
    def ae_err_one(x):                      # x: 1-D numpy array
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
            return float(((m(t) - t) ** 2).mean())
else:
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=2000, random_state=seed).fit(Xtr[ytr == 0], Xtr[ytr == 0])
    def ae_err_one(x): return float(((mlp.predict(x[None, :]) - x[None, :]) ** 2).mean())

xgb = make_xgb(seed).fit(Xtr, ytr)
def logit_one(x):
    p = float(np.clip(xgb.predict_proba(x[None, :])[0, 1], 1e-6, 1 - 1e-6)); return float(np.clip(math.log(p / (1 - p)), -10, 10))

# train-only standardisation constants (as in the experiments)
la = np.array([math.log(ae_err_one(x) + 1e-12) for x in Xtr]); lx = np.array([logit_one(x) for x in Xtr])
mu_a, sd_a, mu_x, sd_x = la.mean(), la.std(), lx.mean(), lx.std(); w = 0.5
def hybrid_one(x):
    return w * (math.log(ae_err_one(x) + 1e-12) - mu_a) / sd_a + (1 - w) * (logit_one(x) - mu_x) / sd_x
cal = np.sort(np.array([hybrid_one(x) for x in Xca[yca == 0]])); nc = len(cal)
thr = cal[min(math.ceil((nc + 1) * (1 - a.alpha)), nc) - 1]

# ---- timing ----
def timeit(fn, label):
    xs = [Xte[i % len(Xte)] for i in range(a.n + a.warmup)]
    for x in xs[:a.warmup]: fn(x)
    t = np.empty(a.n)
    for k, x in enumerate(xs[a.warmup:]):
        t0 = time.perf_counter_ns(); fn(x); t[k] = (time.perf_counter_ns() - t0) / 1e6
    return dict(stage=label, mean_ms=t.mean(), median_ms=np.median(t), p95_ms=np.percentile(t, 95),
                p99_ms=np.percentile(t, 99), max_ms=t.max(), n=a.n)

pool = list(cal); state = {"alpha_t": a.alpha}
def adaptive_one(x):
    s = hybrid_one(x)
    arr = np.sort(np.array(pool[-a.window:])); n = len(arr)
    at = min(max(state["alpha_t"], 1 / (n + 1)), 0.5)
    t = arr[min(math.ceil((n + 1) * (1 - at)), n) - 1]
    flag = s > t
    state["alpha_t"] += 0.05 * (a.alpha - flag)      # update as if the sample were confirmed normal
    pool.append(s)
    return flag
rows = [timeit(ae_err_one, "Autoencoder error only"), timeit(logit_one, "XGBoost probability only"),
        timeit(lambda x: hybrid_one(x) > thr, "Hybrid score + static conformal decision"),
        timeit(adaptive_one, "Hybrid score + adaptive threshold update")]
R = pd.DataFrame(rows).round(4); print(R.to_string(index=False))
import os; os.makedirs(a.out, exist_ok=True); R.to_csv(f"{a.out}/latency_hybrid_conformal.csv", index=False)
print("\nSingle thread, batch size 1, CPU. Excludes data collection (Prometheus scrape interval).")
