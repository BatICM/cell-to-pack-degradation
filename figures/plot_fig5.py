from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.spatial import ConvexHull
import shap
params = {
    "font.family": "serif",
    "font.serif": "Arial",
    "font.style": "normal",
    "font.weight": "normal",
    "font.size": 10,
}
rcParams.update(params)

# In[] Fig5a_1
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SOURCE_XLSX = BBASE_DIR.parent / "source_data" / "Fig5_source_data.xlsx"

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig5a_1")

df["Bin"] = pd.to_numeric(df["Bin"], errors="coerce")
df["Mean absolute SHAP"] = pd.to_numeric(df["Mean absolute SHAP"], errors="coerce")
df = df.dropna(subset=["Feature label", "Bin", "Mean absolute SHAP"])

heatmap_data = (
    df.pivot(index="Feature label", columns="Bin", values="Mean absolute SHAP")
    .reindex(index=["F1", "F2", "F3"])
)

colors = ['#90A3D1', '#87B9C4', '#85C7C2']
n_bins = 100
cmap_blue_cyan = LinearSegmentedColormap.from_list(
    "blue_cyan",
    colors,
    N=n_bins
)

plt.figure(figsize=(2.1, 0.7))

ax = sns.heatmap(
    heatmap_data,
    cmap=cmap_blue_cyan,
    annot=False,
    cbar_kws={"shrink": 1.0},
    yticklabels=["F1", "F2", "F3"],
    linewidths=1.0
)

plt.xticks(rotation=0, ha="right", fontsize=10)
plt.yticks(fontsize=10)

colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=8)

ax.set_xticks([])
ax.set_xlabel("")
ax.set_ylabel("")

plt.show()
# In[] Fig5a_2

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig5a_2")

df["Bin"] = pd.to_numeric(df["Bin"], errors="coerce")
df["Mean absolute SHAP"] = pd.to_numeric(df["Mean absolute SHAP"], errors="coerce")
df = df.dropna(subset=["Feature label", "Bin", "Mean absolute SHAP"])

heatmap_data = (
    df.pivot(index="Feature label", columns="Bin", values="Mean absolute SHAP")
    .reindex(index=["F1", "F2", "F3"])
)

colors = ['#90A3D1', '#87B9C4', '#85C7C2']
n_bins = 100
cmap_blue_cyan = LinearSegmentedColormap.from_list(
    "blue_cyan",
    colors,
    N=n_bins
)

plt.figure(figsize=(2.1, 0.7))

ax = sns.heatmap(
    heatmap_data,
    cmap=cmap_blue_cyan,
    annot=False,
    cbar_kws={"shrink": 1.0},
    yticklabels=["F1", "F2", "F3"],
    linewidths=1.0
)

plt.xticks(rotation=0, ha="right", fontsize=10)
plt.yticks(fontsize=10)

colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=8)

ax.set_xticks([])
ax.set_xlabel("")
ax.set_ylabel("")

plt.show()
# In[] Fig5a_3

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig5a_3")

df["Bin"] = pd.to_numeric(df["Bin"], errors="coerce")
df["Mean absolute SHAP"] = pd.to_numeric(df["Mean absolute SHAP"], errors="coerce")
df = df.dropna(subset=["Feature label", "Bin", "Mean absolute SHAP"])

heatmap_data = (
    df.pivot(index="Feature label", columns="Bin", values="Mean absolute SHAP")
    .reindex(index=["F1", "F2", "F3"])
)

colors = ['#90A3D1', '#87B9C4', '#85C7C2']
n_bins = 100
cmap_blue_cyan = LinearSegmentedColormap.from_list(
    "blue_cyan",
    colors,
    N=n_bins
)

plt.figure(figsize=(2.1, 0.7))

ax = sns.heatmap(
    heatmap_data,
    cmap=cmap_blue_cyan,
    annot=False,
    cbar_kws={"shrink": 1.0},
    yticklabels=["F1", "F2", "F3"],
    linewidths=1.0
)

plt.xticks(rotation=0, ha="right", fontsize=10)
plt.yticks(fontsize=10)

colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=8)

ax.set_xticks([])
ax.set_xlabel("")
ax.set_ylabel("")

plt.show()
# In[] Fig5a_4

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig5a_4")

df["Bin"] = pd.to_numeric(df["Bin"], errors="coerce")
df["Mean absolute SHAP"] = pd.to_numeric(df["Mean absolute SHAP"], errors="coerce")
df = df.dropna(subset=["Feature label", "Bin", "Mean absolute SHAP"])

heatmap_data = (
    df.pivot(index="Feature label", columns="Bin", values="Mean absolute SHAP")
    .reindex(index=["F1", "F2", "F3"])
)

colors = ['#90A3D1', '#87B9C4', '#85C7C2']
n_bins = 100
cmap_blue_cyan = LinearSegmentedColormap.from_list(
    "blue_cyan",
    colors,
    N=n_bins
)

plt.figure(figsize=(2.1, 0.7))

ax = sns.heatmap(
    heatmap_data,
    cmap=cmap_blue_cyan,
    annot=False,
    cbar_kws={"shrink": 1.0},
    yticklabels=["F1", "F2", "F3"],
    linewidths=1.0
)

plt.xticks(rotation=0, ha="right", fontsize=12)
plt.yticks(fontsize=10)

colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=8)

ax.set_xticks([])
ax.set_xlabel("")
ax.set_ylabel("")

plt.show()
# In[] Fig5(b)

colors = ["#90A3D1", "#85C7C2", "#ff7f0e"]
cmap_blue_cyan = LinearSegmentedColormap.from_list("blue_cyan", colors, N=30)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig5b")

df["Sample"] = pd.to_numeric(df["Sample"], errors="coerce")
df["Feature order"] = pd.to_numeric(df["Feature order"], errors="coerce")
df["SHAP value"] = pd.to_numeric(df["SHAP value"], errors="coerce")
df["|Feature value|"] = pd.to_numeric(df["|Feature value|"], errors="coerce")

df = df.dropna(subset=[
    "Sample",
    "Feature",
    "Feature order",
    "SHAP value",
    "|Feature value|"
])

df = df.sort_values(["Sample", "Feature order"])

feature_info = (
    df[["Feature order", "Feature"]]
    .drop_duplicates()
    .sort_values("Feature order")
)

feature_names = feature_info["Feature"].tolist()

shap_matrix = (
    df.pivot(index="Sample", columns="Feature", values="SHAP value")
    .reindex(columns=feature_names)
    .to_numpy()
)

x_matrix = (
    df.pivot(index="Sample", columns="Feature", values="|Feature value|")
    .reindex(columns=feature_names)
)

fig, ax = plt.subplots(figsize=(2.1, 1.2), dpi=600)

shap.summary_plot(
    shap_matrix,
    x_matrix,
    plot_type="dot",
    cmap=cmap_blue_cyan,
    show=False,
    plot_size=None,
    color_bar=False,
)

ax = plt.gca()
ax.set_xlabel("SHAP value (impact on model output)", fontsize=8)
ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=10)

sm = ScalarMappable(cmap=cmap_blue_cyan, norm=Normalize(vmin=0, vmax=1))
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label("|Feature value|", fontsize=8, rotation=270, labelpad=-5)
cbar.set_ticks([0, 1])
cbar.ax.set_yticklabels(["Low", "High"], fontsize=8)

ax.yaxis.set_label_position("left")
ax.yaxis.set_tick_params(pad=-10)

plt.subplots_adjust(left=0.15)
plt.show()
# In[] Fig5(c)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig5c")

df["t-SNE1"] = pd.to_numeric(df["t-SNE1"], errors="coerce")
df["t-SNE2"] = pd.to_numeric(df["t-SNE2"], errors="coerce")
df["Feature code"] = pd.to_numeric(df["Feature code"], errors="coerce")

df = df.dropna(subset=["t-SNE1", "t-SNE2", "Feature code"])

colors = ['#FB9792', '#85C7C2', '#3E8EC4']
n_bins = 100
cmap_blue_cyan = LinearSegmentedColormap.from_list("blue_cyan", colors, N=n_bins)

palette = {
    0: colors[0],
    1: colors[1],
    2: colors[2],
}

markers = {
    0: "o",
    1: "s",
    2: "^",
}

plt.figure(figsize=(2.1, 1.5), dpi=600)

for cls in sorted(df["Feature code"].unique()):
    cls = int(cls)
    sub = df[df["Feature code"] == cls]

    x = sub["t-SNE1"].to_numpy()
    y = sub["t-SNE2"].to_numpy()
    points = np.column_stack([x, y])

    plt.scatter(
        x,
        y,
        c=palette[cls],
        label=f"F{cls + 1}",
        s=15,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.2,
        marker=markers[cls],
        zorder=2
    )

    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            poly = points[hull.vertices]

            plt.fill(
                poly[:, 0],
                poly[:, 1],
                facecolor=palette[cls],
                alpha=0.10,
                zorder=1
            )

            plt.plot(
                np.r_[poly[:, 0], poly[0, 0]],
                np.r_[poly[:, 1], poly[0, 1]],
                color=palette[cls],
                linewidth=1.0,
                alpha=0.7,
                zorder=3
            )
        except Exception:
            pass

plt.xlabel("t-SNE1", fontsize=12)
plt.ylabel("t-SNE2", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.45, 1.35),
    ncol=3,
    frameon=False,
    fontsize=10
)

plt.show()
# In[] Fig5(d)
df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig5d")

df["Inconsistency Metric (%²)"] = pd.to_numeric(df["Inconsistency Metric (%²)"], errors="coerce")
df["Improved (%)"] = pd.to_numeric(df["Improved (%)"], errors="coerce")
df = df.dropna(subset=["Dataset", "Group", "Inconsistency Metric (%²)", "Improved (%)"])

df_plot = df[df["Used in plot"] == True].copy()

dataset_order = ["Data1", "Data2"]

fig, axes = plt.subplots(1, 2, figsize=(1.4 * 4, 1.6))

for ax, dataset_name, cmap, ylim, yticks in zip(
    axes,
    dataset_order,
    ["Blues", "Oranges"],
    [(0, 10), (0, 50)],
    [[0, 3, 6, 9], [0, 15, 30, 45]]
):
    sub_data = df_plot[df_plot["Dataset"] == dataset_name]
    all_imp = []

    for group in sub_data["Group"].drop_duplicates():
        sub = sub_data[sub_data["Group"] == group]

        q = sub["Inconsistency Metric (%²)"].to_numpy()
        imp = sub["Improved (%)"].to_numpy()

        if len(q) > 1 and len(imp) > 1:
            all_imp.extend(imp)

            try:
                sns.kdeplot(
                    x=q,
                    y=imp,
                    fill=True,
                    cmap=cmap,
                    thresh=0.05,
                    ax=ax,
                    alpha=0.8
                )
            except Exception:
                ax.scatter(q, imp, s=6, alpha=0.6)

    if len(all_imp) > 0:
        avg = np.mean(all_imp)
        ax.axhline(avg, color="red", linestyle="--", linewidth=1.2)

        x_min, x_max = ax.get_xlim()
        ax.text(
            x_max * 0.95,
            avg,
            f"Avg: {avg:.2f}",
            color="red",
            fontsize=10,
            va="bottom",
            ha="right"
        )

    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

fig.text(0.5, -0.14, "Heterogeneity Metric (%²)", ha="center", fontsize=12)
fig.text(0.04, 0.5, "Improved (%)", va="center", rotation="vertical", fontsize=12)

plt.show()
