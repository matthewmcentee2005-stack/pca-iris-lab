import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score, LeaveOneOut

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200,
})
SPECIES = ["I. setosa", "I. versicolor", "I. virginica"]
COLOURS = ["#1f5fa8", "#e07b20", "#3c3c3c"]
MARKERS = ["o", "s", "^"]
SHORT = ["Sepal length", "Sepal width", "Petal length", "Petal width"]


def standardise(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def covariance_matrix(x):
    xc = x - x.mean(axis=0)
    return xc.T @ xc / len(x)


def covariance_loop(x):
    n = x.shape[1]
    k = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k[i, j] = np.sum((x[:, i] - x[:, i].mean()) * (x[:, j] - x[:, j].mean())) / len(x)
    return k


def sorted_eigen(k):
    # eigh used instead of eig (symmetric matrix, real output); eigenvectors are columns
    vals, vecs = np.linalg.eigh(k)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    signs = np.sign(vecs[np.argmax(np.abs(vecs), axis=0), np.arange(vecs.shape[1])])
    return vals, vecs * signs


def project(z, vecs):
    return z @ vecs


def bootstrap(x, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    vals_all, ratio_all, vec_all = [], [], []
    _, ref = sorted_eigen(covariance_matrix(standardise(x)))
    for _ in range(n_boot):
        xb = x[rng.integers(0, len(x), len(x))]
        vals, vecs = sorted_eigen(covariance_matrix(standardise(xb)))
        vecs *= np.sign(np.sum(vecs * ref, axis=0))  # match signs to the full-data eigenvectors
        vals_all.append(vals); ratio_all.append(vals / vals.sum()); vec_all.append(vecs)
    return np.array(vals_all), np.array(ratio_all), np.array(vec_all)


def resolution_test(x, n_mc=2000, half_width=0.05, seed=2):
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_mc):
        xn = x + rng.uniform(-half_width, half_width, x.shape)
        vals, _ = sorted_eigen(covariance_matrix(standardise(xn)))
        ratios.append(vals / vals.sum())
    return np.array(ratios)


data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
X = df.to_numpy()
y = data.target
print(df.describe().round(2))

fig, ax = plt.subplots(4, 4, figsize=(7.0, 7.0))
for r in range(4):
    for c in range(4):
        a = ax[r, c]
        if c > r:
            a.axis("off")
            continue
        if r == c:
            bins = np.linspace(X[:, c].min(), X[:, c].max(), 16)
            for s in range(3):
                a.hist(X[y == s, c], bins=bins, color=COLOURS[s], alpha=0.6)
            a.set_yticks([])
        else:
            for s in range(3):
                a.scatter(X[y == s, c], X[y == s, r], s=7, c=COLOURS[s], marker=MARKERS[s])
            if c == 0:
                a.set_ylabel(SHORT[r] + " (cm)", fontsize=9)
            else:
                a.set_yticklabels([])
        if r == 3:
            a.set_xlabel(SHORT[c] + " (cm)", fontsize=9)
        else:
            a.set_xticklabels([])
        a.tick_params(labelsize=8)
handles = [plt.Line2D([], [], ls="", marker=MARKERS[s], color=COLOURS[s], markersize=7, label=SPECIES[s]) for s in range(3)]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.93, 0.80), frameon=False, fontsize=11)
fig.suptitle("Iris measurements: each variable against the others", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(f"{OUT}/fig_pairplot.png"); plt.close(fig)


Z = standardise(X)
print("standardised means:", Z.mean(axis=0).round(12), " sigma:", Z.std(axis=0))


K_raw = covariance_matrix(X)
K = covariance_matrix(Z)
assert np.allclose(K, covariance_loop(Z)) and np.allclose(K_raw, covariance_loop(X))
assert np.allclose(K, K.T)
print("raw covariance (cm^2):\n", K_raw.round(3))
print("standardised covariance (correlation):\n", K.round(3))


vals_eig, vecs_eig = np.linalg.eig(K)
print("np.linalg.eig dtype:", vals_eig.dtype, "order returned:", vals_eig.round(3))
vals, vecs = sorted_eigen(K)
print("unit length check:", np.linalg.norm(vecs, axis=0))
print("orthogonality check (V^T V = I):", np.allclose(vecs.T @ vecs, np.eye(4)))
ratio = vals / vals.sum()
print("eigenvalues:", vals.round(4), " sum:", vals.sum().round(6))
print("explained variance ratio:", ratio.round(4), " cumulative:", np.cumsum(ratio).round(4))
print("eigenvectors (columns = PC1..PC4):\n", vecs.round(3))

vals_raw, vecs_raw = sorted_eigen(K_raw)
print("UNSTANDARDISED eigenvalues (cm^2):", vals_raw.round(4), " ratio:", (vals_raw / vals_raw.sum()).round(4))
print("UNSTANDARDISED PC1:", vecs_raw[:, 0].round(3))


P = project(Z, vecs)
print("variance of projected columns:", P.var(axis=0).round(4))
print("correlation of PC scores (should be identity):\n", np.corrcoef(P.T).round(10))
for s in range(3):
    print(f"{SPECIES[s]:14s} PC1 mean {P[y == s, 0].mean():6.2f} sd {P[y == s, 0].std():.2f} "
          f"range [{P[y == s, 0].min():.2f}, {P[y == s, 0].max():.2f}] | "
          f"PC2 mean {P[y == s, 1].mean():6.2f} sd {P[y == s, 1].std():.2f}")


Zs = StandardScaler().fit_transform(X)
skl = PCA(n_components=4).fit(Zs)
print("sklearn ratio:", skl.explained_variance_ratio_.round(4))
print("sklearn explained_variance_ (N-1):", skl.explained_variance_.round(4), " x149/150:",
      (skl.explained_variance_ * 149 / 150).round(4))
comp = skl.components_.T
print("max |difference| in loadings (up to sign):", np.abs(np.abs(comp) - np.abs(vecs)).max())
print("max |difference| in scores (up to sign):",
      np.abs(np.abs(skl.transform(Zs)) - np.abs(P)).max())


bv, br, bvec = bootstrap(X)
print("bootstrap eigenvalue sd:", bv.std(axis=0).round(4))
print("bootstrap ratio mean:", br.mean(axis=0).round(4), " sd:", br.std(axis=0).round(4))
print("bootstrap cumulative sd:", np.cumsum(br, axis=1).std(axis=0).round(4))
print("bootstrap PC1 loading sd:", bvec[:, :, 0].std(axis=0).round(3))
print("bootstrap PC2 loading sd:", bvec[:, :, 1].std(axis=0).round(3))
rr = resolution_test(X)
print("resolution MC ratio sd:", rr.std(axis=0).round(5), " mean shift:", (rr.mean(axis=0) - ratio).round(5))

lo, hi = P[y == 2, 0].min(), P[y == 1, 0].max()
n_over = np.sum((P[:, 0] >= lo) & (P[:, 0] <= hi) & (y > 0))
print(f"PC1 overlap region [{lo:.2f}, {hi:.2f}] contains {n_over} versicolor/virginica flowers")
print("gap setosa max -> versicolor min on PC1:", round(P[y == 1, 0].min() - P[y == 0, 0].max(), 2))


loo = LeaveOneOut()
for name, feats in [("4 standardised variables", Z), ("PC1 + PC2", P[:, :2]), ("PC1 only", P[:, :1]),
                    ("sepal length + width only", Z[:, :2])]:
    acc = cross_val_score(NearestCentroid(), feats, y, cv=loo).mean()
    print(f"nearest-centroid LOO accuracy, {name}: {acc:.3f}")


fig, ax = plt.subplots(figsize=(6.0, 3.6))
pcs = np.arange(1, 5)
ax.bar(pcs, 100 * ratio, yerr=100 * br.std(axis=0), capsize=4, color="#9a9a9a",
       edgecolor="black", width=0.6, label="Individual PC")
ax.errorbar(pcs, 100 * np.cumsum(ratio), yerr=100 * np.cumsum(br, axis=1).std(axis=0),
            fmt="o-", color=COLOURS[0], capsize=4, label="Cumulative")
ax.axhline(95, color="black", ls="--", lw=0.8)
ax.text(0.55, 92, "95%", va="top", fontsize=9)
for i in range(4):
    ax.text(pcs[i] + 0.33, 100 * ratio[i] / 2 if i == 0 else 100 * ratio[i] + 2.5,
            f"{100 * ratio[i]:.1f}%", fontsize=9)
ax.set_xticks(pcs); ax.set_xticklabels([f"PC{i}" for i in pcs])
ax.set_xlim(0.5, 4.6); ax.set_ylim(0, 105)
ax.set_xlabel("Principal component"); ax.set_ylabel("Variance explained (%)")
ax.set_title("Scree plot of the standardised Iris data")
ax.legend(frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_scree.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
for s in range(3):
    ax[0].scatter(P[y == s, 0], P[y == s, 1], s=14, c=COLOURS[s], marker=MARKERS[s], label=SPECIES[s])
ax[0].set_xlabel(f"PC1 ({100 * ratio[0]:.1f}% of variance)")
ax[0].set_ylabel(f"PC2 ({100 * ratio[1]:.1f}% of variance)")
ax[0].set_title("(a) Data projected onto PC1 and PC2")
ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.22, 1.0), fontsize=9, handletextpad=0.2)
pad = 0.3
ax[0].set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
ax[0].set_ylim(P[:, 1].min() - pad, P[:, 1].max() + 1.1)
for i in range(4):
    ax[1].annotate("", xy=(vecs[i, 0], vecs[i, 1]), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="->", lw=1.6, color=COLOURS[0]))
offsets = [(0.05, -0.04), (0.06, 0.04), (0.04, 0.07), (0.04, -0.11)]
for i in range(4):
    ax[1].text(vecs[i, 0] + offsets[i][0], vecs[i, 1] + offsets[i][1], SHORT[i], fontsize=9,
               ha="left")
ax[1].axhline(0, color="black", lw=0.6); ax[1].axvline(0, color="black", lw=0.6)
ax[1].set_xlim(-0.45, 1.05); ax[1].set_ylim(-0.3, 1.05)
ax[1].set_xlabel("Loading on PC1"); ax[1].set_ylabel("Loading on PC2")
ax[1].set_title("(b) Loadings")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pcs.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.0, 3.2))
w = 0.38
xs = np.arange(4)
ax.bar(xs - w / 2, vecs_raw[:, 0], w, color="#bdbdbd", edgecolor="black",
       label=f"Unstandardised ({100 * vals_raw[0] / vals_raw.sum():.1f}% of variance)")
ax.bar(xs + w / 2, vecs[:, 0], w, color=COLOURS[0], edgecolor="black",
       label=f"Standardised ({100 * ratio[0]:.1f}% of variance)")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(SHORT)
ax.set_ylabel("PC1 loading (unitless)"); ax.set_ylim(-0.4, 1.12)
ax.set_title("Composition of PC1 with and without standardisation")
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_standardisation.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.1))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")
W, H = 1.8, 1.4
top = [(0.1, "1. Load data\nload_iris()\n150 × 4 (cm)"), (2.1, "2. Explore\npair plots,\nhistograms"),
       (4.1, "3. Standardise\nEqn. 4\nmean 0, σ = 1"), (6.1, "4. Covariance\nmatrix K\nEqn. 5 (4 × 4)"),
       (8.1, "5. Eigenvectors\neigh(K), sorted\nby eigenvalue")]
bottom = [(8.1, "6. Project\nZ · V\n(PC scores)"), (5.6, "7. Check with\nscikit-learn\nPCA()"),
          (3.1, "8. Bootstrap\nsteps 3–6 on\n2000 resamples"), (0.6, "9. Plot and\ninterpret\nresults")]
for x0, t in top:
    ax.add_patch(plt.Rectangle((x0, 2.4), W, H, fill=False, lw=1.2))
    ax.text(x0 + W / 2, 2.4 + H / 2, t, ha="center", va="center", fontsize=9)
for x0, t in bottom:
    ax.add_patch(plt.Rectangle((x0, 0.2), W, H, fill=False, lw=1.2))
    ax.text(x0 + W / 2, 0.2 + H / 2, t, ha="center", va="center", fontsize=9)
arrow = dict(arrowstyle="->", lw=1.1)
for i in range(4):
    ax.annotate("", xy=(top[i + 1][0], 3.1), xytext=(top[i][0] + W, 3.1), arrowprops=arrow)
ax.annotate("", xy=(9.0, 0.2 + H), xytext=(9.0, 2.4), arrowprops=arrow)
for i in range(3):
    ax.annotate("", xy=(bottom[i + 1][0] + W, 0.9), xytext=(bottom[i][0], 0.9), arrowprops=arrow)
ax.text(5.0, 4.28, "Steps of the PCA program", ha="center", va="top", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_flowchart.png"); plt.close(fig)

print("figures written to", OUT)
