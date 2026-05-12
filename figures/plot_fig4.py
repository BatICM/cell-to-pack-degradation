from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
rcParams.update({
    "font.family": "serif",
    "font.serif": "Arial",
    "font.size": 12,
})
# In[] Fig4(a)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

SOURCE_XLSX = BASE_DIR / "Fig4_source_data.xlsx"
soh_labels = ["100-95", "95-90", "90-85", "85-80", "80-75"]

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig4a")

df["SOH (%)"] = pd.to_numeric(df["SOH (%)"], errors="coerce")
df["Absolute error (%)"] = pd.to_numeric(df["Absolute error (%)"], errors="coerce")
df = df.dropna(subset=["SOH range", "Absolute error (%)"])

df["SOH range"] = pd.Categorical(
    df["SOH range"],
    categories=soh_labels,
    ordered=True
)

error_stats = (
    df.groupby("SOH range", observed=False)["Absolute error (%)"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

plt.figure(figsize=(2.3, 1.8), dpi=600)

plt.bar(
    error_stats["SOH range"],
    error_stats["mean"],
    yerr=error_stats["std"],
    capsize=4,
    color="#1f77b4",
    alpha=0.4,
    error_kw=dict(
        elinewidth=0.8,
        capthick=1,
        ecolor="black",
        capsize=4
    )
)

plt.ylabel("Absolute error (%)", fontsize=12)
plt.ylim(0, 3)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, alpha=0.3)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.show()

print(error_stats)
# In[] Fig4(b)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig4b")

for col in ["Absolute error (%)", "SOC difference (%)", "SOH difference (%)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Absolute error (%)", "SOC difference (%)", "SOH difference (%)"])

all_errors = df["Absolute error (%)"].to_numpy()
soc_dif = df["SOC difference (%)"].to_numpy()
soh_dif = df["SOH difference (%)"].to_numpy()

colors = ["#90A3D1", "#85C7C2", "#ff7f0e"]
cmap_blue_cyan = LinearSegmentedColormap.from_list("blue_cyan", colors, N=30)

plt.figure(figsize=(2.4, 2.0), dpi=600)

hexbin = plt.hexbin(
    soh_dif,
    soc_dif,
    C=all_errors,
    gridsize=30,
    cmap=cmap_blue_cyan,
    reduce_C_function=np.mean
)

cbar = plt.colorbar(hexbin, label="Mean Absolute Error")
cbar.set_label("Mean Absolute Error", fontsize=8)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)

plt.xlabel("SOH Difference")
plt.ylabel("SOC Difference")
plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

radius_bins = [2, 4, 6, 8, 10]
angle_sectors = 8

for r in radius_bins:
    circle = Circle((0, 0), r, fill=False, color="gray",
                    linestyle="--", linewidth=0.8, alpha=0.7)
    ax.add_patch(circle)

max_radius = radius_bins[-1]
for i in range(angle_sectors):
    angle = 2 * np.pi * i / angle_sectors
    x_end = max_radius * np.cos(angle)
    y_end = max_radius * np.sin(angle)
    ax.plot([0, x_end], [0, y_end], color="gray",
            linestyle="--", linewidth=0.8, alpha=0.7)

radius = np.sqrt(soh_dif**2 + soc_dif**2)
angle = np.arctan2(soc_dif, soh_dif)
angle = np.where(angle < 0, angle + 2 * np.pi, angle)

for r_min, r_max in zip([0] + radius_bins[:-1], radius_bins):
    for i in range(angle_sectors):
        angle_min = 2 * np.pi * i / angle_sectors
        angle_max = 2 * np.pi * (i + 1) / angle_sectors

        mask = (
            (radius >= r_min) & (radius < r_max) &
            (angle >= angle_min) & (angle < angle_max)
        )

        if np.sum(mask) > 0:
            avg_error = np.mean(all_errors[mask])

            center_angle = (angle_min + angle_max) / 2
            center_radius = (r_min + r_max) / 2
            center_x = center_radius * np.cos(center_angle)
            center_y = center_radius * np.sin(center_angle)

            ax.text(
                center_x, center_y, f"{avg_error:.1f}",
                fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1",
                          facecolor="none", edgecolor="none", alpha=0.9)
            )

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xticks([0, 2, 4, 6, 8, 10])
plt.yticks([0, 2, 4, 6, 8, 10])

plt.show()
# In[] Fig4(c)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig4c")

df["Absolute error (%)"] = pd.to_numeric(df["Absolute error (%)"], errors="coerce")
df["Relative error (%)"] = pd.to_numeric(df["Relative error (%)"], errors="coerce")
df = df.dropna(subset=["Absolute error (%)", "Relative error (%)"])

all_errors = df["Absolute error (%)"].to_numpy()
all_relative_errors = df["Relative error (%)"].to_numpy()

sorted_errors = np.sort(all_errors)
sorted_relative_errors = np.sort(all_relative_errors)

cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
cdf_relative = np.arange(1, len(sorted_relative_errors) + 1) / len(sorted_relative_errors)

confidence_level = 0.95
idx_95 = np.argmax(cdf >= confidence_level)
error_95 = sorted_errors[idx_95]

idx_95_relative = np.argmax(cdf_relative >= confidence_level)
error_95_relative = sorted_relative_errors[idx_95_relative]

plt.figure(figsize=(2, 2), dpi=600)

plt.axhline(
    y=0.95,
    color="#ff7f0e",
    linestyle="--",
    alpha=0.7,
    label="95%"
)

plt.plot(
    sorted_errors,
    cdf,
    linewidth=1.5,
    color="#1f77b4",
    label="Absolute error"
)

plt.plot(
    error_95,
    confidence_level,
    marker="o",
    color="#1f77b4",
    markersize=5
)

plt.axvline(
    x=error_95,
    color="gray",
    linestyle="--",
    alpha=0.7
)

plt.xlabel("SOH Error (%)", fontsize=12)
plt.ylabel("Cumulative distribution", fontsize=12)
plt.legend(loc="lower right", fontsize=10, frameon=False)

plt.grid(True, alpha=0.3)
plt.xlim(-0.1, max(sorted_relative_errors) * 1.05)
plt.ylim(0, 1.02)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.show()
# In[] Fig4(d)

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig4d")

cycle = pd.to_numeric(df["Cycle"], errors="coerce").to_numpy()

cell_cols = [
    c for c in df.columns
    if c.startswith("Cell ") and "Test" in c and c.endswith("SOH (%)")
]

weakest = pd.to_numeric(df["Weakest-cell SOH (%)"], errors="coerce").to_numpy()
real = pd.to_numeric(df["Real SOH (%)"], errors="coerce").to_numpy()
proposed = pd.to_numeric(df["Proposed SOH (%)"], errors="coerce").to_numpy()

idx = np.arange(len(df))[4::5]
x = cycle[idx]

plt.figure(figsize=(3, 2), dpi=300)

for col in cell_cols:
    y = pd.to_numeric(df[col], errors="coerce").to_numpy()
    y_sampled = y[idx]

    y_filled = (
        pd.DataFrame(y_sampled)
        .interpolate(method="linear", axis=0, limit_direction="both")
        .values
        .ravel()
    )

    plt.plot(
        x,
        y_filled,
        alpha=1.0,
        color="#D5E9D5",
        linewidth=1.0,
        solid_capstyle="round"
    )

plt.plot(
    x,
    weakest[idx],
    label="Weakest-cell",
    linewidth=2,
    color="#64BDA1",
    linestyle="--"
)

plt.plot(
    x,
    real[idx],
    label="Real",
    linewidth=2,
    color="#ff7f0e",
    alpha=0.8
)

plt.plot(
    x,
    proposed[idx],
    label="Proposed",
    linewidth=1.5,
    color="#1f77b4",
    linestyle="--",
    alpha=0.9
)

plt.legend(loc="upper right", fontsize=8, frameon=False)
plt.xlabel("Cycle", fontsize=12)
plt.ylabel("SOH (%)", fontsize=12)

x_ticks = np.arange(0, len(df), 100)
y_ticks = np.arange(70, 105, 10)

plt.xticks(x_ticks, fontsize=12)
plt.yticks(y_ticks, fontsize=12)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

plt.show()
# In[]

df = pd.read_excel(SOURCE_XLSX, sheet_name="Fig4e")

for col in ["Real SOH (%)", "Estimated SOH (%)", "Error (%)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[
    "Dataset",
    "Target",
    "Real SOH (%)",
    "Estimated SOH (%)",
    "Error (%)"
])


def create_smooth_fill_region(x, y, n_bins=25, smooth_factor=0.5, percentile_range=(5, 95)):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    n_bins = min(n_bins, len(x) // 3)
    n_bins = max(5, n_bins)

    bins = np.linspace(x.min(), x.max(), n_bins)
    digitized = np.digitize(x, bins)

    x_centers, y_mins, y_maxs = [], [], []

    for i in range(1, len(bins)):
        mask = digitized == i

        if np.sum(mask) >= 2:
            y_bin = y[mask]
            x_centers.append((bins[i - 1] + bins[i]) / 2)
            y_mins.append(np.percentile(y_bin, percentile_range[0]))
            y_maxs.append(np.percentile(y_bin, percentile_range[1]))

    x_centers = np.asarray(x_centers)
    y_mins = np.asarray(y_mins)
    y_maxs = np.asarray(y_maxs)

    if len(x_centers) > 3:
        window = max(3, len(x_centers) // 8)

        y_mins_smooth = (
            pd.Series(y_mins)
            .rolling(window=window, center=True)
            .mean()
            .bfill()
            .ffill()
            .to_numpy()
        )

        y_maxs_smooth = (
            pd.Series(y_maxs)
            .rolling(window=window, center=True)
            .mean()
            .bfill()
            .ffill()
            .to_numpy()
        )

        y_mins = smooth_factor * y_mins_smooth + (1 - smooth_factor) * y_mins
        y_maxs = smooth_factor * y_maxs_smooth + (1 - smooth_factor) * y_maxs

    return x_centers, y_mins, y_maxs


def add_error_inset(ax, errors):
    ax_inset = inset_axes(
        ax,
        width="35%",
        height="35%",
        loc="lower right",
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax.transAxes
    )

    errors = np.asarray(errors, dtype=float)

    if len(errors) > 3:
        kde = gaussian_kde(errors)
        x_range = np.linspace(errors.min() * 1.2, errors.max() * 1.2, 200)
        density = kde(x_range)

        ax_inset.fill_between(
            x_range,
            density,
            alpha=0.6,
            color="steelblue"
        )

        ax_inset.plot(
            x_range,
            density,
            color="darkblue",
            linewidth=1.5
        )

    ax_inset.axvline(
        x=0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8
    )

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    std = np.std(errors)

    ax_inset.text(
        -1.6,
        2.5,
        f"RMSE: {rmse:.2f} %\nMAE: {mae:.2f} %\nStd: {std:.2f}",
        transform=ax_inset.transAxes,
        fontsize=8,
        va="top"
    )

    ax_inset.set_ylabel("Density", fontsize=8)
    ax_inset.set_xlabel("Error", fontsize=8)
    ax_inset.xaxis.set_label_position("top")
    ax_inset.xaxis.tick_top()
    ax_inset.tick_params(axis="both", labelsize=7)


colors = ["#96CEB4", "#45B7D1", "#4ECDC4", "#FF6B6B"]
dataset_order = ["Data 102", "Data 105", "Data 302", "Data 4"]
target_order = ["Proposed", "Min-cell"]
ylabel_map = {
    "Proposed": "Proposed (%)",
    "Min-cell": "Min cell (%)",
}

fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.0))
plt.subplots_adjust(wspace=0.2)

for ax, target in zip(axes, target_order):
    target_df = df[df["Target"] == target]

    errors_all = []

    for i, dataset in enumerate(dataset_order):
        sub = target_df[target_df["Dataset"] == dataset]

        if sub.empty:
            continue

        x = sub["Real SOH (%)"].to_numpy()
        y = sub["Estimated SOH (%)"].to_numpy()
        errors = sub["Error (%)"].to_numpy()
        errors_all.extend(errors)

        x_centers, y_mins, y_maxs = create_smooth_fill_region(
            x,
            y,
            n_bins=min(30, len(sub) // 5),
            smooth_factor=0.5,
            percentile_range=(5, 95)
        )

        if len(x_centers) > 0:
            ax.fill_between(
                x_centers,
                y_mins,
                y_maxs,
                alpha=0.8,
                color=colors[i % len(colors)],
                label=dataset
            )

            ax.plot(
                x_centers,
                y_mins,
                color=colors[i % len(colors)],
                linewidth=1,
                alpha=0.8
            )

            ax.plot(
                x_centers,
                y_maxs,
                color=colors[i % len(colors)],
                linewidth=1,
                alpha=0.8
            )

    ax.plot(
        [73, 100],
        [73, 100],
        "k--",
        alpha=0.5,
        linewidth=1,
        label="y=x"
    )

    ax.set_xlabel("Real (%)", fontsize=12)
    ax.set_ylabel(ylabel_map[target], fontsize=12)

    ax.set_xlim(71, 100)
    ax.set_ylim(71, 100)
    ax.set_xticks(np.arange(70, 101, 10))
    ax.set_yticks(np.arange(70, 101, 10))
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(errors_all) > 0:
        add_error_inset(ax, errors_all)

axes[-1].spines["left"].set_visible(False)
axes[-1].spines["right"].set_visible(True)
axes[-1].yaxis.tick_right()
axes[-1].yaxis.set_label_position("right")

plt.show()
# In[] Fig4f

new_order = [
    "SOH_Error_se", "SOH_Error_do", "SOH_Error_detasoc", "SOH_Error_dr", "SOH_Error_to",
    "SOC_Error_se", "SOC_Error_do", "SOC_Error_detasoc", "SOC_Error_dr", "SOC_Error_to",
    "General_Error_se", "General_Error_do", "General_Error_detasoc", "General_Error_dr", "General_Error_to",
]

df1_clean = pd.read_excel(SOURCE_XLSX, sheet_name="Fig4f")
df1_clean = df1_clean.reindex(columns=new_order)

fig, ax = plt.subplots(1, 1, figsize=(9, 2.4))

color = [
    "#D66056", "#CDC7C9", "#ADBBC8", "#7B8BBC", "#424EAD",
    "#D66056", "#CDC7C9", "#ADBBC8", "#7B8BBC", "#424EAD",
    "#D66056", "#CDC7C9", "#ADBBC8", "#7B8BBC", "#424EAD",
]

ax = sns.violinplot(
    data=df1_clean,
    inner=None,
    linewidth=0,
    palette=color,
    ax=ax
)

formatter = FormatStrFormatter("%.1f")
ax.yaxis.set_major_formatter(formatter)

violins = [
    child for child in ax.get_children()
    if isinstance(child, matplotlib.collections.PolyCollection)
]

for i, column in enumerate(df1_clean.columns):
    values = df1_clean[column].dropna()

    if len(values) > 0 and i < len(violins):
        mae = np.mean(values)
        std_dev = np.std(values)

        path = violins[i].get_paths()[0]
        verts = path.vertices

        mae_verts = verts[np.abs(verts[:, 1] - mae) < 0.1]
        lower_verts = verts[np.abs(verts[:, 1] - (mae - std_dev)) < 0.1]
        upper_verts = verts[np.abs(verts[:, 1] - (mae + std_dev)) < 0.1]

        violin_width_mae = mae_verts[:, 0].max() - mae_verts[:, 0].min() if len(mae_verts) > 0 else 0.8
        violin_width_lower = lower_verts[:, 0].max() - lower_verts[:, 0].min() if len(lower_verts) > 0 else 0.8
        violin_width_upper = upper_verts[:, 0].max() - upper_verts[:, 0].min() if len(upper_verts) > 0 else 0.8

        ax.plot(
            [i - violin_width_mae / 2, i + violin_width_mae / 2],
            [mae, mae],
            "k-",
            lw=1
        )

        ax.plot(
            [i - violin_width_lower / 2, i + violin_width_lower / 2],
            [mae - std_dev, mae - std_dev],
            "r-",
            lw=1
        )

        ax.plot(
            [i - violin_width_upper / 2, i + violin_width_upper / 2],
            [mae + std_dev, mae + std_dev],
            "r-",
            lw=1
        )

ax.set_ylabel("Absolute error (%)", fontsize=12)
ax.set_xlabel("")
ax.set_xticks(range(len(df1_clean.columns)))
ax.set_xticklabels([""] * len(df1_clean.columns))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
plt.show()
# In[]Fig4(f)

new_order = [
    "SOH_Error_se", "SOH_Error_do", "SOH_Error_detasoc", "SOH_Error_dr", "SOH_Error_to",
    "SOC_Error_se", "SOC_Error_do", "SOC_Error_detasoc", "SOC_Error_dr", "SOC_Error_to",
    "General_Error_se", "General_Error_do", "General_Error_detasoc", "General_Error_dr", "General_Error_to",
]

df1_clean = pd.read_excel(SOURCE_XLSX, sheet_name="Fig4f")
df1_clean = df1_clean.reindex(columns=new_order)

fig, ax = plt.subplots(1, 1, figsize=(9, 2.4))

color = [
    "#D66056", "#CDC7C9", "#ADBBC8", "#7B8BBC", "#424EAD",
    "#D66056", "#CDC7C9", "#ADBBC8", "#7B8BBC", "#424EAD",
    "#D66056", "#CDC7C9", "#ADBBC8", "#7B8BBC", "#424EAD",
]

ax = sns.violinplot(
    data=df1_clean,
    inner=None,
    linewidth=0,
    palette=color,
    ax=ax
)

formatter = FormatStrFormatter("%.1f")
ax.yaxis.set_major_formatter(formatter)

violins = [
    child for child in ax.get_children()
    if isinstance(child, matplotlib.collections.PolyCollection)
]

for i, column in enumerate(df1_clean.columns):
    values = df1_clean[column].dropna()

    if len(values) > 0 and i < len(violins):
        mae = np.mean(values)
        std_dev = np.std(values)

        path = violins[i].get_paths()[0]
        verts = path.vertices

        mae_verts = verts[np.abs(verts[:, 1] - mae) < 0.1]
        lower_verts = verts[np.abs(verts[:, 1] - (mae - std_dev)) < 0.1]
        upper_verts = verts[np.abs(verts[:, 1] - (mae + std_dev)) < 0.1]

        violin_width_mae = mae_verts[:, 0].max() - mae_verts[:, 0].min() if len(mae_verts) > 0 else 0.8
        violin_width_lower = lower_verts[:, 0].max() - lower_verts[:, 0].min() if len(lower_verts) > 0 else 0.8
        violin_width_upper = upper_verts[:, 0].max() - upper_verts[:, 0].min() if len(upper_verts) > 0 else 0.8

        ax.plot(
            [i - violin_width_mae / 2, i + violin_width_mae / 2],
            [mae, mae],
            "k-",
            lw=1
        )

        ax.plot(
            [i - violin_width_lower / 2, i + violin_width_lower / 2],
            [mae - std_dev, mae - std_dev],
            "r-",
            lw=1
        )

        ax.plot(
            [i - violin_width_upper / 2, i + violin_width_upper / 2],
            [mae + std_dev, mae + std_dev],
            "r-",
            lw=1
        )

ax.set_ylabel("Absolute error (%)", fontsize=12)
ax.set_xlabel("")
ax.set_xticks(range(len(df1_clean.columns)))
ax.set_xticklabels([""] * len(df1_clean.columns))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
plt.show()

