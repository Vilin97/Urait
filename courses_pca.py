"""Perform PCA on course embeddings and visualize the 2D projection."""

#%%
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import patheffects as pe

#%%
# Load course embeddings, apply PCA to 2 dimensions, print explained variance, and visualize.
courses_csv="data/generated/courses.csv"
out_png="data/generated/course_embeddings_pca.png"

df = pd.read_csv(courses_csv)
# Ensure embedding column is parsed from JSON strings if necessary
if df["embedding"].dtype == object:
    df["embedding"] = df["embedding"].apply(lambda s: np.array(json.loads(s), dtype=np.float32))

embeddings = np.vstack(df["embedding"].values)
print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

pca = PCA(n_components=2)
proj = pca.fit_transform(embeddings)

evr = pca.explained_variance_ratio_
print(f"Explained variance ratio (per component): {evr}")
print(f"Explained variance captured by 2 components: {evr.sum():.4f}")

#%%
# Plot
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.figure(figsize=(12, 6))
sc = plt.scatter(proj[:, 0], proj[:, 1], s=20, c=proj[:, 0], cmap="Spectral", alpha=0.8)
plt.colorbar(sc, label="PC1 value")
plt.title("Course embeddings PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")

annotate_top = 30
rng = np.random.default_rng(3)
idxs = rng.choice(proj.shape[0], size=annotate_top, replace=False)

# Highlight annotated points: white outline + filled colored marker
plt.scatter(
    proj[idxs, 0],
    proj[idxs, 1],
    s=140,
    facecolors="none",
    edgecolors="white",
    linewidths=2.0,
    alpha=0.95,
    zorder=3,
)
plt.scatter(
    proj[idxs, 0],
    proj[idxs, 1],
    s=60,
    c="gold",
    edgecolors="black",
    linewidths=0.8,
    alpha=0.95,
    zorder=4,
)

for i in idxs:
    name = df.iloc[i].get("project_name", "")[:80]
    plt.annotate(
        name.lower(),
        (proj[i, 0], proj[i, 1]),
        fontsize=8,
        color="black",
        alpha=0.95,
        xytext=(4, 4),
        textcoords="offset points",
        zorder=5,
        path_effects=[pe.Stroke(linewidth=1.5, foreground="white"), pe.Normal()],
    )

plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"Saved PCA plot to {out_png}")
# Also show in interactive sessions
try:
    plt.show()
except Exception:
    pass
