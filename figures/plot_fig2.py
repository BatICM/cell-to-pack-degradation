import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 10,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

# In[] Fig2(a)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SOURCE_XLSX = BBASE_DIR.parent / "source_data" / "Fig2_source_data.xlsx"

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig2a")

df["Point"] = pd.to_numeric(df["Point"], errors="coerce")
df["SOH (%)"] = pd.to_numeric(df["SOH (%)"], errors="coerce")
df["E (%)"] = pd.to_numeric(df["E (%)"], errors="coerce")
df = df.dropna(subset=["Series", "Geometry", "Point", "SOH (%)", "E (%)"])

plt.figure(figsize=(4.8, 4.3), dpi=600)

zone = df[df["Series"] == "Operating zone"].sort_values("Point")
plt.fill(
    zone["SOH (%)"],
    zone["E (%)"],
    color="#85C7C2",
    alpha=0.3,
    zorder=0
)

line_styles = {
    "Reference line": {"color": "black", "linestyle": "-.", "linewidth": 1.5, "alpha": 1.0},
    "Min ED line": {"color": "#0070C0", "linestyle": "--", "linewidth": 2, "alpha": 0.6},
    "Min EC line": {"color": "#EA700D", "linestyle": "--", "linewidth": 2, "alpha": 0.6},
    "Min SOH line": {"color": "#64BDA1", "linestyle": "--", "linewidth": 2, "alpha": 0.6},
    "Max SOH line": {"color": "#A05195", "linestyle": "--", "linewidth": 2, "alpha": 0.6},
}

for name, style in line_styles.items():
    sub = df[df["Series"] == name].sort_values("Point")
    plt.plot(
        sub["SOH (%)"],
        sub["E (%)"],
        zorder=1,
        **style
    )

cells = df[df["Series"] == "Standard cells"]
plt.scatter(
    cells["SOH (%)"],
    cells["E (%)"],
    s=40,
    facecolors="#FEF1CD",
    edgecolors="black",
    zorder=2,
    alpha=0.6
)

point_styles = {
    "Pack state": {"facecolors": "red", "s": 80, "zorder": 3},
    "Min ED": {"facecolors": "#0070C0", "s": 60, "zorder": 6},
    "Min SOH": {"facecolors": "#4ECDC4", "s": 60, "zorder": 6},
    "Min EC": {"facecolors": "#EA700D", "s": 60, "zorder": 6},
    "Max SOH": {"facecolors": "#A05195", "s": 60, "zorder": 6},
}

for name, style in point_styles.items():
    sub = df[df["Series"] == name]
    plt.scatter(
        sub["SOH (%)"],
        sub["E (%)"],
        marker="o",
        edgecolors="black",
        alpha=0.6,
        **style
    )

legend_elements1 = [
    Line2D([0], [0], marker="o", color="w", label=r"Min. $E_{\mathrm{D}}$",
           markerfacecolor="#0070C0", markersize=8, markeredgecolor="black", alpha=0.6),
    Line2D([0], [0], marker="o", color="w", label=r"Min. SOH",
           markerfacecolor="#4ECDC4", markersize=8, markeredgecolor="black", alpha=0.6),
    Line2D([0], [0], marker="o", color="w", label=r"Min. $E_{\mathrm{C}}$",
           markerfacecolor="#EA700D", markersize=8, markeredgecolor="black", alpha=0.6),
    Line2D([0], [0], marker="o", color="w", label=r"Max. SOH",
           markerfacecolor="#A05195", markersize=8, markeredgecolor="black", alpha=0.6),
    Line2D([0], [0], color="#85C7C2", alpha=0.3, lw=10,
           label=r"$\mathcal{S}_{E,Q}$"),
]

legend_elements2 = [
    Line2D([0], [0], marker="o", color="w", label="Standard cells",
           markerfacecolor="#FEF1CD", markersize=8, markeredgecolor="black"),
    Line2D([0], [0], marker="o", color="w", label="Pack state",
           markerfacecolor="red", markersize=8, markeredgecolor="black", alpha=0.6),
]

legend1 = plt.legend(
    handles=legend_elements1,
    loc="upper left",
    bbox_to_anchor=(0.00, 1.02),
    frameon=False,
    fontsize=10
)

legend2 = plt.legend(
    handles=legend_elements2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.10),
    ncol=2,
    frameon=False,
    fontsize=10
)

plt.gca().add_artist(legend1)

plt.xlim(91.2, 97.8)
plt.ylim(91.2, 97.8)

plt.xticks([91.5, 93.0, 94.5, 96.0, 97.5], fontsize=12)
plt.yticks([91.5, 93.0, 94.5, 96.0, 97.5], fontsize=12)

plt.xlabel("SOH (%)", fontsize=12)
plt.ylabel("E = SOC * SOH (%)", fontsize=12)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

plt.tight_layout()
plt.show()
# In[] Fig2(b)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig2b")
df["Relative error (%)"] = pd.to_numeric(df["Relative error (%)"], errors="coerce")
df = df.dropna(subset=["Series", "Relative error (%)"])

colors = {
    "52-Series": "#2E86C1",
    "16-Series": "#C0392B",
}

fig, ax = plt.subplots(figsize=(2.4, 1.2))

sns.histplot(
    data=df,
    x="Relative error (%)",
    hue="Series",
    bins=20,
    kde=False,
    element="step",
    fill=True,
    alpha=0.25,
    palette=colors,
    linewidth=1.5,
    common_norm=False,
    legend=False,
    ax=ax
)

legend_elements = [
    Patch(
        facecolor=colors["52-Series"],
        alpha=0.4,
        edgecolor=colors["52-Series"],
        linewidth=1.5,
        label="52-Series"
    ),
    Patch(
        facecolor=colors["16-Series"],
        alpha=0.4,
        edgecolor=colors["16-Series"],
        linewidth=1.5,
        label="16-Series"
    ),
]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    frameon=False,
    fontsize=7,
    borderaxespad=0
)

mean_52 = df.loc[df["Series"] == "52-Series", "Relative error (%)"].mean()
mean_16 = df.loc[df["Series"] == "16-Series", "Relative error (%)"].mean()

ymin, ymax = ax.get_ylim()

ax.vlines(
    mean_52,
    0,
    ymax * 0.90,
    colors=colors["52-Series"],
    linestyles="--",
    linewidth=1.5
)
ax.text(
    mean_52,
    ymax * 0.98,
    f"{mean_52:.2f}",
    color=colors["52-Series"],
    ha="center",
    fontsize=8
)

ax.vlines(
    mean_16,
    0,
    ymax * 0.75,
    colors=colors["16-Series"],
    linestyles="--",
    linewidth=1.5
)
ax.text(
    mean_16,
    ymax * 0.77,
    f"{mean_16:.2f}",
    color=colors["16-Series"],
    ha="center",
    fontsize=8
)

ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

sns.despine(ax=ax, offset=0)

ax.set_xlabel("Relative Error (%)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.tick_params(axis="both", which="major", labelsize=9)
ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

plt.show()

print(f"52-Series: Mean={mean_52:.3f}%, Std={df.loc[df['Series'] == '52-Series', 'Relative error (%)'].std():.3f}%")
print(f"16-Series: Mean={mean_16:.3f}%, Std={df.loc[df['Series'] == '16-Series', 'Relative error (%)'].std():.3f}%")
# In[] Fig2(c)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig2c")

df["Heterogeneity index (%²)"] = pd.to_numeric(df["Heterogeneity index (%²)"], errors="coerce")
df["Lack Capacity (%)"] = pd.to_numeric(df["Lack Capacity (%)"], errors="coerce")
df = df.dropna(subset=["Heterogeneity index (%²)", "Lack Capacity (%)"])

colors = ['#FFFFFF', '#ff7f0e', '#90A3D1', '#85C7C2']
cmap_blue_cyan = LinearSegmentedColormap.from_list('blue_cyan', colors, N=60)

g = sns.jointplot(
    x='Heterogeneity index (%²)',
    y='Lack Capacity (%)',
    data=df,
    kind='hex',
    height=2.6,
    ratio=3,
    cmap=cmap_blue_cyan,
    gridsize=40,
    marginal_kws=dict(bins=50, fill=False)
)

x = df["Heterogeneity index (%²)"].to_numpy()
y = df["Lack Capacity (%)"].to_numpy()

g.ax_marg_x.clear()
g.ax_marg_y.clear()

kde_x = gaussian_kde(x)
kde_y = gaussian_kde(y)

x_range = np.linspace(x.min(), x.max(), 200)
y_range = np.linspace(y.min(), y.max(), 200)

kde_values_x = kde_x(x_range)
kde_values_y = kde_y(y_range)

norm_x = (kde_values_x - kde_values_x.min()) / (kde_values_x.max() - kde_values_x.min())
norm_y = (kde_values_y - kde_values_y.min()) / (kde_values_y.max() - kde_values_y.min())

for i in range(len(x_range) - 1):
    color = cmap_blue_cyan(norm_x[i])
    g.ax_marg_x.fill_between(
        x_range[i:i+2],
        kde_values_x[i:i+2],
        0,
        color=color,
        alpha=0.8,
        edgecolor='none'
    )

for i in range(len(y_range) - 1):
    color = cmap_blue_cyan(norm_y[i])
    g.ax_marg_y.fill_betweenx(
        y_range[i:i+2],
        kde_values_y[i:i+2],
        0,
        color=color,
        alpha=0.8,
        edgecolor='none'
    )

g.ax_marg_x.set_xlim(g.ax_joint.get_xlim())
g.ax_marg_y.set_ylim(g.ax_joint.get_ylim())

sns.regplot(
    x='Heterogeneity index (%²)',
    y='Lack Capacity (%)',
    data=df,
    ax=g.ax_joint,
    scatter=False,
    color='red',
    line_kws={'linewidth': 1.5, 'linestyle': '--'}
)

g.ax_joint.set_xlim(0, 50)
g.ax_joint.set_ylim(1, 9)
plt.show()

