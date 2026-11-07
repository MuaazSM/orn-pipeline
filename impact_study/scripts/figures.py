"""Render CT/label figures for the deck from figdata.npz + results.json."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PID = sys.argv[1]
res = json.load(open("results.json"))
d = np.load("figdata.npz")
A = d[f"{PID}_affine"]; sp = d[f"{PID}_spacing"]
names = json.load(open("subvolume_names.json"))  # class -> label, colour

BG = "#FAF9F5"
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "text.color": "#3F4743"})


def radiological(sl):
    """Array [i, j] -> display rows=anterior..posterior, cols=patient right..left."""
    out = sl.T  # rows j, cols i
    if A[0, 0] > 0: out = out[:, ::-1]
    if A[1, 1] > 0: out = out[::-1, :]
    return out


def coronal(pr):
    """Projection [i, k] -> rows superior..inferior, cols patient right..left."""
    out = pr.T
    if A[0, 0] > 0: out = out[:, ::-1]
    if A[2, 2] > 0: out = out[::-1, :]
    return out


def bbox(m, pad):
    idx = np.argwhere(m > 0)
    lo = np.maximum(idx.min(0) - pad, 0); hi = np.minimum(idx.max(0) + pad + 1, m.shape)
    return lo, hi


def cmap_for(table, n):
    cols = ["#00000000"] + [table.get(str(k), {"color": "#999999"})["color"] for k in range(1, n + 1)]
    return ListedColormap(cols)


def save(fig, name):
    fig.savefig(f"fig_{name}.png", dpi=220, facecolor=BG, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# 1) Axial CT with mandible sub-volumes, and axial CT with teeth
for key, lab_key, table, n in [("axial", "mand", names["mandible"], 6),
                               ("axial_teeth", "teeth", names["teeth"], 32)]:
    img = d[f"{PID}_{key}_img" if key == "axial" else f"{PID}_axial_teeth_img"].astype(float)
    lab = d[f"{PID}_axial_{lab_key}"]
    img_d, lab_d = radiological(img), radiological(lab)
    lo, hi = bbox(lab_d, 28)
    img_d, lab_d = img_d[lo[0]:hi[0], lo[1]:hi[1]], lab_d[lo[0]:hi[0], lo[1]:hi[1]]
    fig, ax = plt.subplots(figsize=(4.2, 4.2 * img_d.shape[0] / img_d.shape[1]))
    ax.imshow(np.clip(img_d, -200, 1600), cmap="gray", vmin=-200, vmax=1600, interpolation="bilinear")
    masked = np.ma.masked_where(lab_d == 0, lab_d)
    ax.imshow(masked, cmap=cmap_for(table, n), vmin=0, vmax=n, alpha=0.55, interpolation="nearest")
    ax.set_axis_off()
    save(fig, f"{key}_{PID}")

# 2) Coronal front-view projections of each label set
for nm, table, n in [("mand", names["mandible"], 6), ("maxi", names["maxilla"], 6), ("teeth", names["teeth"], 32)]:
    pr = coronal(d[f"{PID}_cor_{nm}"])
    lo, hi = bbox(pr, 6)
    pr = pr[lo[0]:hi[0], lo[1]:hi[1]]
    aspect = sp[2] / sp[0]
    h, w = pr.shape[0] * aspect, pr.shape[1]
    fig, ax = plt.subplots(figsize=(4.0, 4.0 * h / w))
    ax.imshow(np.ma.masked_where(pr == 0, pr), cmap=cmap_for(table, n), vmin=0, vmax=n,
              aspect=aspect, interpolation="nearest")
    ax.set_axis_off()
    save(fig, f"coronal_{nm}_{PID}")
print("figures done")
