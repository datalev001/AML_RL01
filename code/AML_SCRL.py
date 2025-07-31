import os, json, warnings, random
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.metrics         import log_loss, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)

# ------------------------------------------------------------
# 0. Load data (no generation here)
# ------------------------------------------------------------
DATA_DIR      = r"..."
PANEL_CSV     = os.path.join(DATA_DIR, "dtr_panel.csv")
SPIKES_CSV    = os.path.join(DATA_DIR, "dtr_spike_events.csv")

panel  = pd.read_csv(PANEL_CSV)
spikes = pd.read_csv(SPIKES_CSV)

print(f"Panel rows  : {len(panel):,}  |  Spike rows  : {len(spikes):,}")
print(panel.head(3))

# ------------------------------------------------------------
# Quick EDA – see what the trainees are dealing with
# ------------------------------------------------------------
# 1‑a.  Action distribution
action_map = {0: "full_limit", 1: "low_limit", 2: "freeze"}
panel['action_name'] = panel['action'].map(action_map)
action_counts = panel['action_name'].value_counts(normalize=True)
print("\nAction distribution:")
print(action_counts.round(3))

# Fraud‑loss distribution per action (boxplot)
plt.figure(figsize=(6,3.5))
panel.boxplot(column='fraud_loss_next_30d', by='action_name', showfliers=False)
plt.suptitle('')
plt.title("Fraud‑loss vs action")
plt.ylabel("USD"); plt.tight_layout()
plt.show()

# Spike‑ratio histogram (just to visualise rarity / skew)
plt.figure(figsize=(5,3))
plt.hist(panel['spike_amount_ratio'], bins=40, alpha=0.7)
plt.title("Spike‑intensity distribution"); plt.xlabel("ratio"); plt.ylabel("count")
plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# Propensity‑score model  e_a(x)   (multinomial logistic)
# ------------------------------------------------------------
FEATURES = [
    'segment', 'income', 'kyc_risk_score', 'geo_risk_score', 'dormant_days',
    'spike_amount_ratio', 'txn_cnt_7d', 'device_change_flag', 'past_fraud_flag'
]
TARGET_TREAT = 'action'

X = panel[FEATURES]
y = panel[TARGET_TREAT]

num_cols = [c for c in FEATURES if panel[c].dtype != 'object']
cat_cols = [c for c in FEATURES if panel[c].dtype == 'object']

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

prop_clf = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, multi_class='multinomial'))
])

prop_clf.fit(X, y)
panel['propensity_a'] = prop_clf.predict_proba(X).max(axis=1)

print("\nMultinomial‑logit training log‑loss:",
      log_loss(y, prop_clf.predict_proba(X)))

# ------------------------------------------------------------
# Outcome regressors  m_a(x)  - one for each action
# ------------------------------------------------------------
outcome_models = {}
for a in (0,1,2):
    mask   = panel['action'] == a
    gbm    = Pipeline([("prep", preprocess),
                       ("gbr", GradientBoostingRegressor(max_depth=3,
                                                         n_estimators=300,
                                                         learning_rate=0.05))])
    gbm.fit(X[mask], panel.loc[mask, 'fraud_loss_next_30d'])
    outcome_models[a] = gbm

    preds = gbm.predict(X[mask])
    mae   = mean_absolute_error(panel.loc[mask,'fraud_loss_next_30d'], preds)
    print(f"GBR action={a}  MAE = {mae:.2f}")

# Store m̂_a(x) predictions for *all* actions (counter‑factuals!)
m_hat = np.zeros((len(panel), 3))
for a in (0,1,2):
    m_hat[:, a] = outcome_models[a].predict(X)

# ------------------------------------------------------------
# IPW‑DR counter‑factual estimate  τ̂(a|x)
#     τ̂ = m̂_a(x) + 1(e=a) * w * (Y - m̂_a(x))
# ------------------------------------------------------------
observed_Y = panel['fraud_loss_next_30d'].values
e_x        = prop_clf.predict_proba(X)                   # shape (N,3)

w = np.zeros((len(panel),3))
for a in (0,1,2):
    w[:,a] = (panel['action']==a).astype(float) / np.clip(e_x[:,a], 1e-3, None)

tau_hat = m_hat + w * (observed_Y.reshape(-1,1) - m_hat)

print("\nCounter‑factual mean fraud‑loss by action (IPW‑DR):")
for a in (0,1,2):
    print(f"  a={a:>1}  loss ≈ {tau_hat[:,a].mean():6.2f} USD")

# ------------------------------------------------------------
# Simple Policy‑Gradient on top of the SCM simulator
#    (Linear softmax policy  πθ  with Lagrange loss‑cap)
# ------------------------------------------------------------
class SoftmaxPolicy(nn.Module):
    def __init__(self, in_dim, n_act=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_act, bias=True)
    def forward(self, x):                     # x already scaled/encoded
        logits = self.linear(x)
        return torch.softmax(logits, dim=1)

# Build a *frozen* design matrix once (scaled / one‑hot)
X_np = preprocess.fit_transform(X).astype(np.float32)
X_tensor = torch.tensor(X_np)

policy = SoftmaxPolicy(X_tensor.shape[1]).train()
optimP = optim.Adam(policy.parameters(), lr=1e-2)

LOSS_CAP = 50.0
lam      = 5.0       # Lagrange multiplier
EPOCHS   = 8
BATCH    = 4096

for epoch in range(EPOCHS):
    idx = np.random.choice(len(panel), BATCH, replace=False)
    xb  = X_tensor[idx]
    # turn m_hat / tau_hat into torch
    tau_b = torch.tensor(tau_hat[idx], dtype=torch.float32)

    pi    = policy(xb)                           # (B,3)
    exp_L = (pi * tau_b).sum(dim=1).mean()       # expected fraud loss
    mild_penalty = (pi[:,2] * 8.0).mean()        # discourage too many freezes
    pg_loss = exp_L + lam * torch.relu(exp_L - LOSS_CAP) + mild_penalty

    optimP.zero_grad(); pg_loss.backward(); optimP.step()

    print(f"Epoch {epoch:2d}   E[L]={exp_L.item():6.2f}   "
          f">50? {exp_L.item() > LOSS_CAP}")

# ------------------------------------------------------------
# Deploy policy on spike events  –– fixed version
# ------------------------------------------------------------
policy.eval()

# a) policy probabilities need the *transformed* matrix
spike_X_num = preprocess.transform(spikes[FEATURES]).astype(np.float32)
spike_pi    = policy(torch.tensor(spike_X_num)).detach().numpy()

# b) loss predictions need the *raw* feature frame (Pipeline handles encoding)
raw_spike_X = spikes[FEATURES]
loss_matrix = np.column_stack([
    outcome_models[a].predict(raw_spike_X) for a in (0, 1, 2)
])                                           # shape (N,3)

# c) expected loss under the new policy
spikes['opt_action'] = spike_pi.argmax(axis=1)
spikes['opt_loss']   = (spike_pi * loss_matrix).sum(axis=1)

print("\nSample recommended actions for first five spike events:")
disp_cols = ['account_id','month_index','spike_amount_ratio',
             'kyc_risk_score','geo_risk_score','opt_action','opt_loss']
print(spikes[disp_cols].head())

# ------------------------------------------------------------
# Visualise policy shift vs historical
# ------------------------------------------------------------
hist_counts = panel['action'].value_counts(normalize=True).sort_index()
new_counts  = spikes['opt_action'].value_counts(normalize=True).sort_index()

plt.figure(figsize=(5,3))
plt.bar(np.arange(3)-0.15, hist_counts, width=0.3, label="historical")
plt.bar(np.arange(3)+0.15, new_counts,  width=0.3, label="policy")
plt.xticks([0,1,2], ['full','low','freeze'])
plt.ylabel("probability"); plt.title("Action mix: historic vs new RL policy")
plt.legend(); plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# Save learned policy coefficients & a quick JSON lookup
# ------------------------------------------------------------
theta = { 'coef': policy.linear.weight.detach().numpy().tolist(),
          'bias': policy.linear.bias.detach().numpy().tolist(),
          'scaler_mean' : preprocess.named_transformers_['num'].mean_.tolist(),
          'scaler_scale': preprocess.named_transformers_['num'].scale_.tolist(),
          'feature_names': preprocess.get_feature_names_out().tolist() }

with open("rl_policy_theta.json", "w") as fh:
    json.dump(theta, fh, indent=2)

print("\nSaved RL policy parameters →  rl_policy_theta.json")
print("=== walkthrough complete ===")