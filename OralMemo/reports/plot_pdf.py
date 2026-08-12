from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "figures" / "case_report_summary.csv"
FIG_DIR = ROOT / "figures"


COLORS = {
    "navy": "#2F5597",
    "blue": "#4E79A7",
    "cyan": "#76B7B2",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "purple": "#B07AA1",
    "brown": "#9C755F",
    "pink": "#FF9DA7",
    "gray": "#6B7280",
    "dark_gray": "#374151",
    "light_gray": "#E5E7EB",
}

PALETTE = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["cyan"],
    COLORS["brown"],
    COLORS["pink"],
    COLORS["gray"],
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 12,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce")
    df["max_duration_months"] = pd.to_numeric(df["max_duration_months"], errors="coerce")
    df["duration_years"] = df["max_duration_months"] / 12.0
    df["field"] = df["field"].fillna("Unknown").replace("", "Unknown")
    df["journal"] = df["journal"].fillna("Unknown").replace("", "Unknown")
    return df


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    print(f"saved {FIG_DIR / (name + '.pdf')}")


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.0f}") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.5, value * 0.015),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#111827",
        )


def plot_publication_years(df: pd.DataFrame) -> None:
    years = df.dropna(subset=["pub_year"]).assign(pub_year=lambda d: d["pub_year"].astype(int))
    year_counts = years["pub_year"].value_counts().sort_index()
    year_counts = year_counts.reindex(range(year_counts.index.min(), year_counts.index.max() + 1), fill_value=0)

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    bars = ax.bar(year_counts.index, year_counts.values, color=COLORS["blue"], alpha=0.82, width=0.72)
    ax.plot(year_counts.index, year_counts.values, color=COLORS["navy"], marker="o", linewidth=2.0, markersize=4.8)
    annotate_bars(ax, bars)
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Case reports")
    ax.set_xticks(year_counts.index[::2])
    ax.set_ylim(0, year_counts.max() * 1.22)
    ax.yaxis.grid(True, color=COLORS["light_gray"], linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "case_report_publication_years")
    plt.close(fig)


def plot_followup_duration(df: pd.DataFrame) -> None:
    duration = df["duration_years"].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    bins = [0, 1, 2, 3, 5, 10, 20, 45]
    counts, _, bars = ax.hist(duration, bins=bins, color=COLORS["orange"], alpha=0.86, edgecolor="white", linewidth=1.0)
    annotate_bars(ax, bars)
    ax.axvline(duration.median(), color=COLORS["red"], linewidth=1.8, label=f"Median {duration.median():.1f} y")
    ax.axvline(duration.mean(), color=COLORS["dark_gray"], linewidth=1.6, linestyle="--", label=f"Mean {duration.mean():.1f} y")
    ax.set_xlabel("Follow-up duration (years)")
    ax.set_ylabel("Case reports")
    ax.set_xticks([0, 5, 10, 20, 30, 40])
    ax.set_ylim(0, max(counts) * 1.18)
    ax.legend(loc="upper right", fontsize=11, handlelength=2.0, borderaxespad=0.5)
    ax.yaxis.grid(True, color=COLORS["light_gray"], linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "case_report_followup_horizon")
    plt.close(fig)


def plot_subfield_distribution(df: pd.DataFrame) -> None:
    counts = df["field"].value_counts()
    major = counts[counts >= 10]
    other = counts[counts < 10].sum()
    if other:
        major = pd.concat([major, pd.Series({"Other": other})])

    total = major.sum()
    fig = plt.figure(figsize=(8.2, 4.5))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.08, 1.0), wspace=0.02)
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])
    wedges, _, autotexts = ax.pie(
        major.values,
        startangle=90,
        counterclock=False,
        colors=PALETTE[: len(major)],
        autopct=lambda pct: f"{int(round(pct * total / 100.0))}",
        pctdistance=0.78,
        radius=1.08,
        textprops={"fontsize": 11, "fontweight": "bold", "color": "#111827"},
        wedgeprops={"width": 0.44, "edgecolor": "white", "linewidth": 1.2},
    )
    for text in autotexts:
        text.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.7})
    ax.text(0, 0.07, str(len(df)), ha="center", va="center", fontsize=25, fontweight="bold", color=COLORS["dark_gray"])
    ax.text(0, -0.14, "reports", ha="center", va="center", fontsize=11, fontweight="bold", color=COLORS["gray"])
    legend_labels = [f"{name}  {count} ({count / len(df):.1%})" for name, count in major.items()]
    legend_ax.axis("off")
    legend_ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        fontsize=11,
        handlelength=1.2,
        handletextpad=0.7,
        labelspacing=0.8,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.97)
    save(fig, "case_report_subfield_composition")
    plt.close(fig)


def plot_top_journals(df: pd.DataFrame) -> None:
    journal_counts = df["journal"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    y_positions = range(len(journal_counts))
    ax.hlines(y_positions, 0, journal_counts.values, color=COLORS["light_gray"], linewidth=3.2)
    ax.scatter(journal_counts.values, y_positions, color=COLORS["purple"], s=88, zorder=3)
    for y, value in zip(y_positions, journal_counts.values):
        ax.text(value + 0.55, y, str(value), va="center", fontsize=11, fontweight="bold", color=COLORS["dark_gray"])
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(journal_counts.index)
    ax.set_xlabel("Case reports")
    ax.set_ylabel("Journal")
    ax.set_xlim(0, journal_counts.max() * 1.18)
    ax.xaxis.grid(True, color=COLORS["light_gray"], linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "case_report_top_journals")
    plt.close(fig)


def remove_legacy_overview() -> None:
    legacy = FIG_DIR / "case_report_collection_overview.pdf"
    if legacy.exists():
        legacy.unlink()


def write_summary(df: pd.DataFrame) -> None:
    duration = df["duration_years"].dropna()
    lines = [
        "# Case Report Collection Statistics",
        "",
        f"- Total retained PDFs: {len(df)}",
        f"- Follow-up duration: min {duration.min():.1f} years, median {duration.median():.1f} years, mean {duration.mean():.1f} years, max {duration.max():.1f} years",
        "",
        "## Subfields",
    ]
    for field, count in df["field"].value_counts().items():
        lines.append(f"- {field}: {count}")
    lines.extend(["", "## Top journals"])
    for journal, count in df["journal"].value_counts().head(20).items():
        lines.append(f"- {journal}: {count}")
    (FIG_DIR / "case_report_collection_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    df = load_data()
    remove_legacy_overview()
    plot_publication_years(df)
    plot_followup_duration(df)
    plot_subfield_distribution(df)
    plot_top_journals(df)
    write_summary(df)


if __name__ == "__main__":
    main()
