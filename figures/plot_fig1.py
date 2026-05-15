from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib import rcParams
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Arial',
    'font.size': 10,
})
# In[]Fig1a
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SOURCE_XLSX = BASE_DIR.parent / "source_data" / "Fig1_source_data.xlsx"

dataset_order = ["ESSL1", "ESSL2", "PBSL1", "ESSH1"]
title_map = {
    "ESSL1": r"$\mathrm{ESS_{L1}}$",
    "ESSL2": r"$\mathrm{ESS_{L2}}$",
    "PBSL1": r"$\mathrm{PBS_{L1}}$",
    "ESSH1": r"$\mathrm{ESS_{H1}}$",
}

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig1a")
df["SOH (%)"] = pd.to_numeric(df["SOH (%)"], errors="coerce")
df = df.dropna(subset=["SOH (%)"])

fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.0), dpi=600)
colors = ["#1f77b4", "#ff7f0e"]

for i, dataset in enumerate(dataset_order):
    ax = axes[i]
    sub = df[df["Dataset"] == dataset]

    c_clean = sub[sub["Level"] == "Cell"]["SOH (%)"].to_numpy(dtype=float)
    p_clean = sub[sub["Level"] == "Pack"]["SOH (%)"].to_numpy(dtype=float)

    all_data = np.concatenate([c_clean, p_clean])
    x_min, x_max = np.percentile(all_data, [0.5, 99.5])
    x_min = np.floor(x_min / 5) * 5
    x_max = 104
    bins = np.linspace(x_min, x_max, 18)

    ax.hist(c_clean, bins=bins, density=True, alpha=0.5,
            color=colors[0], edgecolor="white", linewidth=0.3, label="Cell")
    ax.hist(p_clean, bins=bins, density=True, alpha=0.6,
            color=colors[1], edgecolor="white", linewidth=0.3, label="Pack")

    x_range = np.linspace(x_min, x_max, 500)
    ax.plot(x_range, gaussian_kde(c_clean)(x_range), color=colors[0], lw=1.5)
    ax.plot(x_range, gaussian_kde(p_clean)(x_range), color=colors[1], lw=1.5)

    ax.set_title(title_map[dataset], fontsize=12)
    ax.invert_xaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x*100:g}"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.tick_params(axis="both", labelsize=10)

    if i == 0:
        ax.set_ylabel("Density (%)", fontsize=12)

fig.text(0.5, 0.01, "SOH (%)", ha="center", fontsize=12)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.95, 0.95),
           frameon=False, fontsize=10, handlelength=1.5)

plt.tight_layout(rect=[0, 0.03, 0.9, 1])
plt.show()
# In[] Fig1(b)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig1b")

df["Throughput (MWh)"] = pd.to_numeric(df["Throughput (MWh)"], errors="coerce")
df["Cumulative throughput (MWh)"] = pd.to_numeric(df["Cumulative throughput (MWh)"], errors="coerce")
df = df.dropna(subset=["Period", "Pack", "Throughput (MWh)", "Cumulative throughput (MWh)"])

periods = df["Period"].drop_duplicates().tolist()

boxplot_data = [
    df.loc[df["Period"] == period, "Throughput (MWh)"].dropna().to_numpy()
    for period in periods
]

mean_cumulative = (
    df.groupby("Period")["Cumulative throughput (MWh)"]
    .mean()
    .reindex(periods)
)

plt.figure(figsize=(5.2, 2.4))

positions = range(1, len(periods) + 1)

boxprops = dict(linestyle="-", linewidth=0.5, color="darkblue")
whiskerprops = dict(linestyle="-", linewidth=0.5, color="black")
capprops = dict(linestyle="-", linewidth=0.5, color="black")
medianprops = dict(linestyle="-", linewidth=0.5, color="red")
flierprops = dict(marker="o", markersize=2)

plt.boxplot(
    boxplot_data,
    positions=positions,
    labels=periods,
    patch_artist=True,
    boxprops=boxprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    medianprops=medianprops,
    flierprops=flierprops,
    widths=0.6
)

ax1 = plt.gca()
ax1.set_ylim(0, 70)
ax1.set_ylabel("Monthly (MWh)", fontsize=12, color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4", labelsize=10)
ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
plt.xticks(fontsize=10, rotation=90)
plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

ax2 = ax1.twinx()
ax2.plot(
    positions,
    mean_cumulative.values,
    marker="o",
    linewidth=1.5,
    markersize=5,
    markerfacecolor="white",
    markeredgecolor="#64AC9A",
    markeredgewidth=1.5,
    color="#64AC9A",
    zorder=4
)

ax2.set_ylabel("Cumulative (MWh)", fontsize=12, color="#64AC9A")
ax2.tick_params(axis="y", labelcolor="#64AC9A", labelsize=10)
ax2.set_ylim(0, 550)
ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

plt.tight_layout()
plt.show()
# In[] Fig1(c)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig1c")
charge_rates = pd.to_numeric(df["Charge rate (C)"], errors="coerce").dropna().to_numpy()

n_bins = 5
min_val = charge_rates.min()
max_val = charge_rates.max()

start_val = np.floor(min_val * 100) / 100
target_width = 0.08

bin_edges = [round(start_val + i * target_width, 2) for i in range(n_bins + 1)]
counts, _ = np.histogram(charge_rates, bins=bin_edges)
percentages = counts / len(charge_rates) * 100

bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(n_bins)]

plt.figure(figsize=(2, 2))
plt.bar(
    range(n_bins),
    percentages,
    color="steelblue",
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5
)

plt.ylabel("Percentage (%)", fontsize=12)
plt.xticks(range(n_bins), bin_labels, rotation=90, fontsize=10)
plt.yticks(fontsize=10)

plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
# In[] Fig1(d)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig1d")

df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce")
df["SOH (%)"] = pd.to_numeric(df["SOH (%)"], errors="coerce")
df = df.dropna(subset=["Dataset", "Level", "Cycle", "SOH (%)"])


def get_series(dataset, level):
    sub = df[(df["Dataset"] == dataset) & (df["Level"] == level)]
    sub = sub.sort_values("Cycle")
    return sub["Cycle"].to_numpy(), sub["SOH (%)"].to_numpy()


essl1_cell_x, essl1_cell_y = get_series("ESSL1", "Cell")
essl1_pack_x, essl1_pack_y = get_series("ESSL1", "Pack")
pbsl1_cell_x, pbsl1_cell_y = get_series("PBSL1", "Cell")
pbsl1_pack_x, pbsl1_pack_y = get_series("PBSL1", "Pack")


plt.figure(figsize=(2.1, 2.3), dpi=600)

plt.plot(
    essl1_cell_x[::20],
    essl1_cell_y[::20],
    color="#1f77b4",
    alpha=0.8,
    label="Cell"
)

plt.plot(
    essl1_pack_x[::20],
    essl1_pack_y[::20],
    color="#1f77b4",
    alpha=0.8,
    linestyle="--",
    label="Pack"
)

plt.plot(
    pbsl1_cell_x[::30],
    pbsl1_cell_y[::30],
    color="#ff7f0e",
    alpha=0.8,
    label="Cell"
)

plt.plot(
    pbsl1_pack_x[::20],
    pbsl1_pack_y[::20],
    color="#ff7f0e",
    alpha=0.8,
    linestyle="--",
    label="Pack"
)

plt.ylim(80, 102)
plt.xlim(0, 600)
plt.xticks([0, 200, 400, 600])

plt.xlabel("Cycle", fontsize=12)
plt.ylabel("SOH (%)", fontsize=12)
plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend(fontsize=8, frameon=False)
plt.show()
# In[] Fig1(e)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig1e")

df["SOH loss (%)"] = pd.to_numeric(df["SOH loss (%)"], errors="coerce")
df = df.dropna(subset=["Dataset", "Level", "Cycle", "SOH loss (%)"])

dataset_order = df["Dataset"].drop_duplicates().tolist()
cycle_order = df["Cycle"].drop_duplicates().tolist()

colors = ["#90A3D1", "#87B9C4", "#85C7C2", "#ff7f0e"]
cmap_blue_cyan = LinearSegmentedColormap.from_list("blue_cyan", colors, N=50)

vmin_global = df["SOH loss (%)"].min()
vmax_global = df["SOH loss (%)"].max()

num_datasets = len(dataset_order)
fig, axes = plt.subplots(
    num_datasets,
    1,
    figsize=(1.8, 0.58 * num_datasets),
    dpi=300
)

if num_datasets == 1:
    axes = [axes]

plt.subplots_adjust(hspace=0.1)

for i, dataset in enumerate(dataset_order):
    ax = axes[i]

    sub = df[df["Dataset"] == dataset]

    data_slice = (
        sub.pivot(index="Level", columns="Cycle", values="SOH loss (%)")
        .reindex(index=["Cell", "Pack"], columns=cycle_order)
    )

    ax.set_facecolor("#E0E0E0")

    sns.heatmap(
        data_slice,
        cmap=cmap_blue_cyan,
        ax=ax,
        yticklabels=["Cell", "Pack"],
        vmin=vmin_global,
        vmax=vmax_global,
        cbar=False,
        linewidths=0.8,
        linecolor="white"
    )

    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8, rotation=0)

    num_cols = data_slice.shape[1]
    ax.set_xticks(np.arange(num_cols) + 0.5)

    if i == num_datasets - 1:
        ax.set_xticklabels(cycle_order, rotation=90, fontsize=10)
    else:
        ax.set_xticklabels([])

sm = plt.cm.ScalarMappable(
    cmap=cmap_blue_cyan,
    norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global)
)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("SOH Loss (%)", fontsize=8)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)

for spine in cbar_ax.spines.values():
    spine.set_visible(False)

plt.show()
# In[]Fig1(f)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig1f")

df["EFC"] = pd.to_numeric(df["EFC"], errors="coerce")
df["Capacity (Ah)"] = pd.to_numeric(df["Capacity (Ah)"], errors="coerce")
df = df.dropna(subset=["EFC", "Metric", "Capacity (Ah)"])

def get_series(metric):
    sub = df[df["Metric"] == metric].sort_values("EFC")
    return sub["EFC"].to_numpy(), sub["Capacity (Ah)"].to_numpy()

efc_pack, cap_pack = get_series("Pack")
efc_weakest, cap_weakest = get_series("Weakest-cell")

plt.figure(figsize=(2.1, 2.3), dpi=600)

plt.plot(
    efc_pack[::48],
    cap_pack[::48],
    "-o",
    label="Pack",
    markersize=3,
    linewidth=1
)

plt.plot(
    efc_weakest[::48],
    cap_weakest[::48],
    "-o",
    label="Weakest-cell",
    markersize=3,
    linewidth=1
)

plt.xlabel("EFC", fontsize=12)
plt.ylabel("Capacity (Ah)", fontsize=12)
plt.xticks([0, 150, 300, 450])

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend(frameon=False, fontsize=8)
plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

plt.show()
# In[]Fig1(g)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig1g")

for col in ["Position", "Voltage (V)", "Left", "Right", "Median (V)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Dataset", "Level", "Axis", "Position", "Voltage (V)", "Left", "Right"])

groups = (
    df[["Dataset", "Position"]]
    .drop_duplicates()
    .sort_values("Position")
)

color_cell = "#1f77b4"
color_pack = "#ff7f0e"
color_map = {
    "Cell": color_cell,
    "Pack": color_pack,
}

fig, ax1 = plt.subplots(figsize=(2, 2.3))
ax2 = ax1.twinx()

for _, item in groups.iterrows():
    dataset = item["Dataset"]

    for level in ["Cell", "Pack"]:
        sub = df[(df["Dataset"] == dataset) & (df["Level"] == level)].copy()

        if sub.empty:
            continue

        sub = sub.sort_values("Voltage (V)")
        ax = ax1 if sub["Axis"].iloc[0] == "left" else ax2

        ax.fill_betweenx(
            sub["Voltage (V)"],
            sub["Left"],
            sub["Right"],
            facecolor=color_map[level],
            alpha=0.7,
            edgecolor=color_map[level],
            linewidth=0
        )

        position = sub["Position"].iloc[0]
        median = sub["Median (V)"].iloc[0]

        ax.hlines(
            median,
            position - 0.6 / 4,
            position + 0.6 / 4,
            colors="white",
            linewidth=1.5,
            zorder=3
        )

ax1.set_xticks(groups["Position"].to_numpy())
ax1.set_xticklabels(groups["Dataset"].tolist(), fontsize=10)

ax1.set_ylim(3.2, 3.70)
ax2.set_ylim(3.0, 4.2)

ax1.set_ylabel("Voltage (V)", fontsize=12)
ax2.set_ylabel("", fontsize=12)

ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)

legend_elements = [
    Patch(facecolor=color_cell, alpha=0.7, edgecolor=color_cell, label="Cell"),
    Patch(facecolor=color_pack, alpha=0.7, edgecolor=color_pack, label="Pack"),
]

ax1.legend(
    handles=legend_elements,
    loc="upper left",
    frameon=False,
    fancybox=True,
    framealpha=0.9,
    fontsize=9
)

plt.show()
# In[] Fig1(h)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig1h")

for col in ["Position", "Voltage (V)", "Left", "Right", "Mean (V)", "Median (V)", "Q25 (V)", "Q75 (V)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Group", "Position", "Voltage (V)", "Left", "Right"])

groups = (
    df[["Group", "Position"]]
    .drop_duplicates()
    .sort_values("Position")
)

colors = ["#90A3D1", "#85C7C2", "#ff7f0e"]

fig, ax = plt.subplots(figsize=(2, 2.4))

means = []

for i, (_, item) in enumerate(groups.iterrows()):
    group = item["Group"]
    position = item["Position"]

    sub = df[df["Group"] == group].sort_values("Voltage (V)")
    color = colors[i % len(colors)]

    ax.fill_betweenx(
        sub["Voltage (V)"],
        sub["Left"],
        sub["Right"],
        facecolor=color,
        alpha=0.7,
        edgecolor=color,
        linewidth=0
    )

    mean_val = sub["Mean (V)"].iloc[0]
    median_val = sub["Median (V)"].iloc[0]
    q25 = sub["Q25 (V)"].iloc[0]
    q75 = sub["Q75 (V)"].iloc[0]

    ax.hlines(median_val, position - 0.6/4, position + 0.6/4,
              colors="white", linewidth=1.5, zorder=3)
    ax.hlines(mean_val, position - 0.6/3, position + 0.6/3,
              colors="white", linewidth=1.0, linestyle="--", zorder=3)
    ax.hlines(q25, position - 0.6/6, position + 0.6/6,
              colors="gray", linewidth=0.5, alpha=0.7, zorder=2)
    ax.hlines(q75, position - 0.6/6, position + 0.6/6,
              colors="gray", linewidth=0.5, alpha=0.7, zorder=2)

    means.append(mean_val)

positions = groups["Position"].to_numpy()
ax.scatter(positions, means, color="white", s=20, zorder=4, edgecolor="black", linewidth=0.5)
ax.plot(positions, means, color="black", linestyle="--", linewidth=1.5, alpha=0.8, zorder=3)

ax.set_xticks(positions)
ax.set_xticklabels(groups["Group"].tolist(), fontsize=10)
ax.set_ylabel("Voltage (V)", fontsize=12)
ax.set_xlabel("Pack structure", fontsize=12)
ax.set_ylim(3.2, 3.55)

ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

ax.spines["left"].set_color("black")
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_color("black")
ax.spines["bottom"].set_linewidth(0.8)
ax.tick_params(axis="both", colors="black")

plt.show()
# In[] Fig(i)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig1i")

num_cols = [
    "module_order", "SOH_percent",
    "deltaV_low_mV", "deltaV_high_mV", "deltaV_fit_mV",
    "deltaT_low_C", "deltaT_high_C", "deltaT_fit_C"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=num_cols)

module_info = (
    df[["module_order", "module"]]
    .drop_duplicates()
    .sort_values("module_order")
)

palette = {
    "ESS$_{L1}$": {"band": "#90A3D1", "line": "#355C7D"},
    "PBS$_{L1}$": {"band": "#F4B183", "line": "#C95F4A"},
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.0, 2.4), sharex=True)

for _, row in module_info.iterrows():
    name = row["module"]
    g = df[df["module"] == name].sort_values("SOH_percent", ascending=False)
    col = palette.get(name, {"band": "#D9D9D9", "line": "#6A6A6A"})

    ax1.fill_between(
        g["SOH_percent"],
        g["deltaV_low_mV"],
        g["deltaV_high_mV"],
        color=col["band"],
        alpha=0.8,
        linewidth=0
    )
    ax1.plot(
        g["SOH_percent"],
        g["deltaV_fit_mV"],
        lw=1.6,
        ls="--",
        color=col["line"],
        label=name
    )

    ax2.fill_between(
        g["SOH_percent"],
        g["deltaT_low_C"],
        g["deltaT_high_C"],
        color=col["band"],
        alpha=0.8,
        linewidth=0
    )
    ax2.plot(
        g["SOH_percent"],
        g["deltaT_fit_C"],
        lw=1.6,
        ls="--",
        color=col["line"],
        label=name
    )

ax1.set_ylabel("ΔV (mV)", fontsize=12)
ax1.tick_params(axis="both", labelsize=10)
ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

ax2.set_ylabel("ΔT (°C)", fontsize=12)
ax2.set_xlabel("SOH (%)", fontsize=12)
ax2.tick_params(axis="both", labelsize=10)
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

ax1.set_xlim(100, 76)
ax2.set_xlim(100, 76)

fig.align_ylabels([ax1, ax2])
plt.show()

