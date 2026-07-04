"""Step 1 of docs/hint_free_repair_direction.md: coefficient predictability.

Can the rank-8 coefficients (projection of the row's hint-delta onto the LOO
PCA basis) be predicted from the row's UNHINTED concept-token states at L30?

Data: focus_state_composite_27b_property_states.npz (13 dev rows, per-position
unhinted states + concept deltas at L30). Outer LOO by row; inner LOO by row
picks the ridge alpha. Gate (pre-stated in the direction doc): predicted-vs-
true coefficient cosine and pooled R^2 meaningfully above a shuffled-pairing
null. Constant baseline = train-mean coefficient vector (the "mean_only"
analogue in coefficient space).
"""
import re
import sys

import numpy as np

sys.path.insert(0, "/scratch/scholar/skiron/phantom-or-real/scripts")
from stage2_rank_k_guard import fit_pca_basis

RANK = 8
SEED = 20260704
N_PERM = 50
ALPHAS = (1e2, 1e3, 1e4, 1e5, 1e6)

npz = np.load(
    "/scratch/scholar/skiron/phantom-or-real/results/stage2/erasure/"
    "focus_state_composite_27b_property_states.npz"
)
rows = sorted({int(m.group(1)) for k in npz.files
               for m in [re.search(r"L30_row(\d+)_", k)] if m})
delta = {r: npz[f"L30_row{r}_concept_delta"].astype(np.float64) for r in rows}
unhinted = {r: npz[f"L30_row{r}_unhinted_concept_states"].astype(np.float64) for r in rows}
print(f"rows: {len(rows)}  positions/row: {[delta[r].shape[0] for r in rows]}")


def ridge_fit(x, y, alpha):
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-8
    xs = (x - mu) / sd
    n, d = xs.shape
    if n <= d:
        # dual form: W = X^T (XX^T + aI)^-1 Y
        k = xs @ xs.T
        w = xs.T @ np.linalg.solve(k + alpha * np.eye(n), y - y.mean(axis=0))
    else:
        w = np.linalg.solve(xs.T @ xs + alpha * np.eye(d), xs.T @ (y - y.mean(axis=0)))
    return mu, sd, w, y.mean(axis=0)


def ridge_predict(model, x):
    mu, sd, w, b = model
    return ((x - mu) / sd) @ w + b


def cos_rows(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def run_fold(test_row, train_rows, y_by_row, perm_rng=None):
    """Returns (per-position cosine, per-position constant-baseline cosine,
    residual variance ratio) for the held-out row."""
    x_tr = np.concatenate([unhinted[r] for r in train_rows], axis=0)
    y_tr = np.concatenate([y_by_row[r] for r in train_rows], axis=0)
    if perm_rng is not None:
        y_tr = y_tr[perm_rng.permutation(len(y_tr))]

    best = None
    for alpha in ALPHAS:
        cs = []
        for held in train_rows:
            inner = [r for r in train_rows if r != held]
            xi = np.concatenate([unhinted[r] for r in inner], axis=0)
            yi = np.concatenate([y_by_row[r] for r in inner], axis=0)
            if perm_rng is not None:
                yi = yi[perm_rng.permutation(len(yi))]
            m = ridge_fit(xi, yi, alpha)
            cs.append(cos_rows(ridge_predict(m, unhinted[held]), y_by_row[held]).mean())
        score = float(np.mean(cs))
        if best is None or score > best[0]:
            best = (score, alpha)
    _, alpha = best

    model = ridge_fit(x_tr, y_tr, alpha)
    y_true = y_by_row[test_row]
    y_pred = ridge_predict(model, unhinted[test_row])
    y_const = np.tile(y_tr.mean(axis=0), (len(y_true), 1))
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_const = ((y_true - y_const) ** 2).sum()
    r2_vs_const = 1.0 - ss_res / (ss_const + 1e-12)
    return cos_rows(y_pred, y_true), cos_rows(y_const, y_true), r2_vs_const, alpha


def full_pass(perm_seed=None):
    cos_pred, cos_const, r2s, alphas = [], [], [], []
    for test_row in rows:
        train_rows = [r for r in rows if r != test_row]
        basis = fit_pca_basis(delta, RANK, exclude_rows={test_row})
        q, mean = basis["components"], basis["mean"]
        y_by_row = {r: (delta[r] - mean) @ q.T for r in rows}
        rng = None if perm_seed is None else np.random.default_rng(perm_seed + test_row)
        cp, cc, r2, alpha = run_fold(test_row, train_rows, y_by_row, rng)
        cos_pred.append(cp); cos_const.append(cc); r2s.append(r2); alphas.append(alpha)
    return (np.concatenate(cos_pred), np.concatenate(cos_const),
            np.array(r2s), alphas)


cp, cc, r2, alphas = full_pass()
print("\n== real model (outer LOO by row, 13 folds) ==")
print(f"mean cosine(pred, true) over {len(cp)} positions: {cp.mean():+.3f}")
print(f"mean cosine(const-mean baseline, true):           {cc.mean():+.3f}")
print(f"per-row R^2 vs constant baseline: mean {r2.mean():+.3f}  "
      f"(rows>0: {(r2 > 0).sum()}/{len(r2)})")
print(f"alphas chosen: {sorted(set(alphas))}")

rng = np.random.default_rng(SEED)
null_means = []
for i in range(N_PERM):
    ncp, _, _, _ = full_pass(perm_seed=int(rng.integers(1, 2**31)))
    null_means.append(ncp.mean())
null_means = np.array(null_means)
ge = int((null_means >= cp.mean()).sum())
print(f"\n== shuffled-pairing null ({N_PERM} permutations) ==")
print(f"null mean cosine: {null_means.mean():+.3f}  range [{null_means.min():+.3f}, {null_means.max():+.3f}]")
print(f"permutations >= observed: {ge}/{N_PERM}")
print(f"\nGATE (doc step 1): observed {cp.mean():+.3f} vs null "
      f"{'PASSES' if ge == 0 and cp.mean() > cc.mean() else 'see numbers'}; "
      f"beats constant baseline: {cp.mean() > cc.mean()}")
