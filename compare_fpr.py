"""Compare the Autoencoder false-alarm rate per seed: train_models_multiseed.py vs conformal script (row A).
Usage: python ml\\compare_fpr.py"""
import pandas as pd, numpy as np
m = pd.read_csv("results/table5_multiseed_per_seed.csv")
m = m[m.model == "Autoencoder"].set_index("seed")["fpr"]
c = pd.read_csv("results/conformal_ablation_per_repeat_table5split_seed00.csv")
c = c[c.method.str.startswith("A")].set_index("rep")["FPR"]
d = pd.DataFrame({"multiseed_script": m, "conformal_script_rowA": c}).dropna()
d["diff"] = d.conformal_script_rowA - d.multiseed_script
print(d.round(3).to_string()); print("\nmeans:", d.mean().round(3).to_dict())
