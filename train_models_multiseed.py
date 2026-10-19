"""
Table 5 over multiple seeds (mean +/- std) with the SAME models/hyper-parameters as train_models.py.
Differences from train_models.py (deliberate, state them in the paper):
  * split + model seed change with SEED (0..N-1) instead of always 42
  * the scaler is fit on the training rows only (no test information)
  * also reports macro-F1, normal-class recall and false-alarm rate (FPR)
Run from the project root:  python ml\\train_models_multiseed.py --seeds 10
Quick check first:          python ml\\train_models_multiseed.py --seeds 2
"""
import argparse, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")
torch.set_num_threads(1)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--csv", default="datasets/processed/microservices_anomaly_dataset.csv")
ap.add_argument("--feature-json", default="results/latency/feature_columns.json")
ap.add_argument("--out", default="results")
a = ap.parse_args()

import json
FEATS = json.load(open(a.feature_json))
df = pd.read_csv(a.csv).replace([np.inf, -np.inf], np.nan).fillna(0)
Xraw = df[FEATS].values.astype(float)
yb = df["binary_label"].values
ymc = LabelEncoder().fit_transform(df["label"].values)

class AE(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.encoder = nn.Sequential(nn.Linear(d, 24), nn.ReLU(), nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 6), nn.ReLU())
        s.decoder = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, d))
    def forward(s, x): return s.decoder(s.encoder(x))

class LSTMAE(nn.Module):
    def __init__(s, d, h=32):
        super().__init__()
        s.encoder = nn.LSTM(d, h, batch_first=True); s.decoder = nn.LSTM(h, d, batch_first=True)
    def forward(s, x): return s.decoder(s.encoder(x)[0])[0]

def train(model, Xn, bs, seed):
    opt = torch.optim.Adam(model.parameters(), lr=0.001); crit = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.tensor(Xn, dtype=torch.float32)), batch_size=bs, shuffle=True, generator=g)
    model.train()
    for _ in range(80):
        for (b,) in dl:
            opt.zero_grad(); crit(model(b), b).backward(); opt.step()
    model.eval(); return model

def errs(model, Z, axes):
    with torch.no_grad(): R = model(torch.tensor(Z, dtype=torch.float32)).numpy()
    return ((Z - R) ** 2).mean(axis=axes)

def bin_metrics(name, yt, yp, seed):
    return dict(seed=seed, model=name, accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"),
                normal_recall=recall_score(yt, yp, pos_label=0), anomaly_recall=recall_score(yt, yp, pos_label=1),
                fpr=1 - recall_score(yt, yp, pos_label=0))

rows = []
for seed in range(a.seeds):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # ---- binary split (70/30, then 80/20 validation) ----
    idx = np.arange(len(yb))
    i_tr, i_te = train_test_split(idx, test_size=0.30, random_state=seed, stratify=yb)
    i_sub, i_val = train_test_split(i_tr, test_size=0.20, random_state=seed, stratify=yb[i_tr])
    sc = StandardScaler().fit(Xraw[i_tr]); Xs = np.nan_to_num(sc.transform(Xraw))
    Xtr_n = Xs[i_sub][yb[i_sub] == 0]; Xval_n = Xs[i_val][yb[i_val] == 0]; Xte, yte = Xs[i_te], yb[i_te]
    # 1 Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=0.20, random_state=seed).fit(Xtr_n)
    rows.append(bin_metrics("Isolation Forest", yte, (iso.predict(Xte) == -1).astype(int), seed))
    # 2 Autoencoder (threshold = 95th pct of validation normals)
    ae = train(AE(Xs.shape[1]), Xtr_n, 16, seed)
    thr = np.percentile(errs(ae, Xval_n, 1), 95)
    rows.append(bin_metrics("Autoencoder", yte, (errs(ae, Xte, 1) > thr).astype(int), seed))
    # 3 LSTM Autoencoder (windows of 5 consecutive rows, as in train_models.py)
    Xq = np.array([Xs[i:i + 5] for i in range(len(Xs) - 4)]); yq = np.array([yb[i + 4] for i in range(len(Xs) - 4)])
    q_tr, q_te = train_test_split(np.arange(len(yq)), test_size=0.30, random_state=seed, stratify=yq)
    q_sub, q_val = train_test_split(q_tr, test_size=0.20, random_state=seed, stratify=yq[q_tr])
    lm = train(LSTMAE(Xs.shape[1]), Xq[q_sub][yq[q_sub] == 0], 8, seed)
    thr_l = np.percentile(errs(lm, Xq[q_val][yq[q_val] == 0], (1, 2)), 95)
    rows.append(bin_metrics("LSTM Autoencoder", yq[q_te], (errs(lm, Xq[q_te], (1, 2)) > thr_l).astype(int), seed))
    # 4 XGBoost binary
    xb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       eval_metric="logloss", random_state=seed).fit(Xs[i_tr], yb[i_tr])
    rows.append(bin_metrics("XGBoost Binary", yte, xb.predict(Xte), seed))
    # 5 XGBoost ten-class
    m_tr, m_te = train_test_split(idx, test_size=0.30, random_state=seed, stratify=ymc)
    sc2 = StandardScaler().fit(Xraw[m_tr])
    xm = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       eval_metric="mlogloss", random_state=seed).fit(sc2.transform(Xraw[m_tr]), ymc[m_tr])
    pm = xm.predict(sc2.transform(Xraw[m_te]))
    rows.append(dict(seed=seed, model="XGBoost Multiclass", accuracy=accuracy_score(ymc[m_te], pm),
                     macro_f1=f1_score(ymc[m_te], pm, average="macro")))
    print(f"seed {seed} done", flush=True)

R = pd.DataFrame(rows); out = Path(a.out); out.mkdir(exist_ok=True)
R.to_csv(out / "table5_multiseed_per_seed.csv", index=False)
g = R.groupby("model", sort=False)[["accuracy", "macro_f1", "normal_recall", "anomaly_recall", "fpr"]].agg(["mean", "std"]).round(4)
g.to_csv(out / "table5_multiseed_summary.csv"); print(g.to_string())
