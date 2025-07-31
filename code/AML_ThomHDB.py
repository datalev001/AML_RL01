import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from collections import defaultdict

CSV_PATH = "aml_bandit_data.csv"
np.random.seed(2025)                 # reproducible Thompson draws

# ---------------------------------------------------------------------
# Data ingestion + quick profiling
# ---------------------------------------------------------------------
alerts = pd.read_csv(CSV_PATH)
print(f"Loaded {len(alerts):,} alerts  •  columns: {list(alerts.columns)}\n")

print("Basic descriptive stats of numerical features\n")
print(alerts[[
    "geo_entropy", "beneficiary_conc",
    "median_tx_gap", "foreign_wire_pct"
]].describe().round(2))

# histogram grid ------------------------------------------------------
fig = plt.figure(figsize=(8, 4))
gs  = gridspec.GridSpec(1, 4)
for i, col in enumerate(
        ["geo_entropy", "beneficiary_conc",
         "median_tx_gap", "foreign_wire_pct"]):
    ax = plt.subplot(gs[i])
    ax.hist(alerts[col], bins=30, alpha=.75)
    ax.set_title(col.replace("_", "\n"))
plt.suptitle("Feature distributions ‑ last 30‑day customer profile")
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------
# HDBSCAN clustering   (daily batch)
# ---------------------------------------------------------------------
FEATURES = [
    "geo_entropy", "beneficiary_conc",
    "median_tx_gap", "foreign_wire_pct"
]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(alerts[FEATURES])

clusterer = HDBSCAN(min_samples=25,
                    min_cluster_size=60,
                    metric="euclidean",
                    prediction_data=True)
cluster_labels = clusterer.fit_predict(X_scaled)
outlier_scores = clusterer.outlier_scores_

alerts["hdb_cluster"]  = cluster_labels          # ‑1 = outlier
alerts["hdb_outlier"]  = (cluster_labels == -1).astype(int)
alerts["hdb_score"]    = np.nan_to_num(outlier_scores)

print("\nHDBSCAN summary:")
print(pd.Series(cluster_labels).value_counts().head())

# ---------------------------------------------------------------------
#  Thompson‑Sampling contextual bandit
#     – 3 actions: q1_immediate / q2_same_day / q3_deferred
# ---------------------------------------------------------------------
ACTIONS  = ["q1_immediate", "q2_same_day", "q3_deferred"]
idx2act  = {i : a for i, a in enumerate(ACTIONS)}
act2idx  = {a : i for i, a in idx2act.items()}

# ► Prior Beta(1,1) = uniform for each arm
alpha = np.ones(3, dtype=float)
beta  = np.ones(3, dtype=float)

# logging helpers
regret_log   = []
sar_count_ts = 0
sar_count_bl = alerts["sar_filed"].sum()       # baseline (historical queue)

for t, row in alerts.iterrows():

    # -------- context -> cluster risk prior  -------------------------
    # Use HDBSCAN results as a single summary context feature:
    # higher outlier score  ->  higher prior risk
    context_risk = row["hdb_score"]
    context_scale = np.clip(context_risk / 3.0, 0, 1)   # simple rescale

    # -------- Thompson draw & action choice --------------------------
    #   Each arm's success‑prob ϑ_a ~ Beta(α_a, β_a)
    sampled_theta = np.random.beta(alpha, beta)
    #   Add tiny context modulation – favour q1 if risk high, q3 if low
    context_bias  = np.array([context_scale,
                              0.0,
                              1.0 - context_scale])
    probs = sampled_theta + 0.05 * context_bias
    a_idx = np.argmax(probs)          # greedy w.r.t sampled θ
    action = idx2act[a_idx]

    # -------- observe reward (ground‑truth) ---------------------------
    reward = int(row["sar_filed"])    # 1 if SAR filed, else 0

    # update Beta posterior
    alpha[a_idx] += reward
    beta[a_idx]  += (1 - reward)

    sar_count_ts += reward

    # ---------- logging ----------------------------------------------
    if (t+1) % 5000 == 0:
        conversion = sar_count_ts / (t+1)
        regret_log.append(conversion)
        print(f"[step {t+1:>5}]   SAR rate (Thompson) = {conversion:.4f}")

print("\nFinal posterior α / β values per arm:")
for i, a in idx2act.items():
    print(f"   {a:<14}  α={alpha[i]:5.1f}  β={beta[i]:5.1f}")

# ---------------------------------------------------------------------
# KPI comparison
# ---------------------------------------------------------------------
rate_ts    = sar_count_ts / len(alerts)
rate_base  = sar_count_bl / len(alerts)

print(f"\n► Historical SAR hit‑rate : {rate_base:.4%}")
print(f"► Thompson SAR hit‑rate   : {rate_ts:.4%}")

lift = (rate_ts - rate_base) / max(rate_base, 1e-6)
print(f"   ==> Relative lift      : {lift:.2%}")

# ---------------------------------------------------------------------
# 5.  Action‑mix plot
# ---------------------------------------------------------------------
hist_mix  = alerts["queue_action"].value_counts(normalize=True)
ts_counts = defaultdict(int)
for a in ACTIONS:
    ts_counts[a] = 0
alpha_tmp = alpha.copy()
beta_tmp  = beta.copy()

# Re‑simulate K draws only for plotting mix (quick & approximate)
K = 20000
for _ in range(K):
    sθ = np.random.beta(alpha_tmp, beta_tmp)
    a  = idx2act[int(np.argmax(sθ))]
    ts_counts[a] += 1
ts_mix = pd.Series({k: v / K for k, v in ts_counts.items()})

fig, ax = plt.subplots(figsize=(5,3))
ax.bar(np.arange(3)-0.15, hist_mix[ACTIONS], width=.3, label="historical")
ax.bar(np.arange(3)+0.15, ts_mix[ACTIONS],   width=.3, label="bandit")
ax.set_xticks(np.arange(3)); ax.set_xticklabels(["q1","q2","q3"])
ax.set_ylabel("probability")
ax.set_title("Queue mix: historic vs Thompson‑sampling policy")
ax.legend(); ax.grid(True, alpha=.3)
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------
# Top‑N priority list demo
# ---------------------------------------------------------------------
alerts["ts_priority"] = alerts.apply(
    lambda r: 1 / (alpha[act2idx["q1_immediate"]] + beta[act2idx["q1_immediate"]])
              * r["hdb_score"], axis=1)

top20 = alerts.sort_values("ts_priority", ascending=False).head(20)
print("\nTop‑20 alerts the policy would feed to analysts first:")
print(top20[["alert_id","customer_id","hdb_score","outlier","ts_priority"]]
      .to_string(index=False))