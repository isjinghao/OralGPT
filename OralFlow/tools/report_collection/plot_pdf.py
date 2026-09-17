from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "report" / "figures"
INPUT_CSV = FIG_DIR / "case_report_summary.csv"
FONT_PATH = PROJECT_ROOT / "iclr26" / "fonts" / "times.ttf"
BOLD_FONT_PATH = PROJECT_ROOT / "iclr26" / "fonts" / "timesbd.ttf"
ITALIC_FONT_PATH = PROJECT_ROOT / "iclr26" / "fonts" / "timesi.ttf"
font_manager.fontManager.addfont(FONT_PATH)
font_manager.fontManager.addfont(BOLD_FONT_PATH)
font_manager.fontManager.addfont(ITALIC_FONT_PATH)
TIMES_NEW_ROMAN = font_manager.FontProperties(fname=FONT_PATH).get_name()
TIMES_NEW_ROMAN_BOLD = font_manager.FontProperties(fname=BOLD_FONT_PATH)
TIMES_NEW_ROMAN_ITALIC = font_manager.FontProperties(fname=ITALIC_FONT_PATH)


COLORS = {
    "blue": "#8EE4F5",
    "blue_dark": "#37C9DF",
    "pink": "#F4C9E7",
    "pink_dark": "#E796CA",
    "gray": "#6B7280",
    "dark_gray": "#111111",
    "light_gray": "#D9D9D9",
}

PALETTE = [COLORS["blue"], COLORS["pink"]]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.size": 14,
            "font.family": TIMES_NEW_ROMAN,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.labelweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce")
    df["max_duration_months"] = pd.to_numeric(df["max_duration_months"], errors="coerce")
    df["duration_years"] = df["max_duration_months"] / 12.0
    df["field"] = df["field"].fillna("Others").replace("", "Others")
    df["journal"] = df["journal"].fillna("Others").replace({"": "Others", "Unknown": "Others", "Unkown": "Others"})
    journal_names = {
        "Medicine (Baltimore)": "Medicine",
        "Clin Adv Periodontics": "Clinical Advances in Periodontics",
        "J Endod": "Journal of Endodontics",
        "Int Endod J": "International Endodontic Journal",
        "Int J Periodontics Restorative Dent": "International Journal of Periodontics and Restorative Dentistry",
        "Head Face Med": "Head and Face Medicine",
        "Oper Dent": "Operative Dentistry",
        "Int J Oral Maxillofac Surg": "International Journal of Oral and Maxillofacial Surgery",
        "Angle Orthod": "The Angle Orthodontist",
        "Oral Surg Oral Med Oral Pathol Oral Radiol": "Oral Surgery, Oral Medicine, Oral Pathology and Oral Radiology",
        "Head Neck Pathol": "Head and Neck Pathology",
        "Clin Implant Dent Relat Res": "Clinical Implant Dentistry and Related Research",
    }
    df["journal"] = df["journal"].replace(journal_names)
    return df


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}")
    print(f"saved {FIG_DIR / (name + '.pdf')}")
    print(f"saved {FIG_DIR / (name + '.svg')}")


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.0f}") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.5, value * 0.015),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=18,
            fontproperties=TIMES_NEW_ROMAN_BOLD,
            color="#111827",
        )


def plot_publication_years(df: pd.DataFrame) -> None:
    years = df.dropna(subset=["pub_year"]).assign(pub_year=lambda d: d["pub_year"].astype(int))
    year_counts = years["pub_year"].value_counts().sort_index()
    year_counts = year_counts.reindex(range(year_counts.index.min(), year_counts.index.max() + 1), fill_value=0)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bars = ax.bar(year_counts.index, year_counts.values, color=COLORS["blue"], width=0.72)
    ax.plot(year_counts.index, year_counts.values, color=COLORS["blue_dark"], marker="o", linewidth=2.2, markersize=5.5)
    annotate_bars(ax, bars)
    ax.set_xlabel("Publication Year", fontsize=18)
    ax.set_ylabel("Case Reports", fontsize=18)
    ax.set_xticks(year_counts.index[::2])
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, year_counts.max() * 1.18)
    ax.yaxis.grid(True, color=COLORS["light_gray"], linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.97)
    save(fig, "case_report_publication_years")
    plt.close(fig)


def plot_followup_duration(df: pd.DataFrame) -> None:
    duration = df["duration_years"].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bins = [0, 1, 2, 3, 5, 10, 20, 45]
    counts, _, bars = ax.hist(duration, bins=bins, color=COLORS["pink"], edgecolor="white", linewidth=1.0)
    annotate_bars(ax, bars)
    ax.axvline(duration.median(), color=COLORS["pink_dark"], linewidth=2.0, label=f"Median {duration.median():.1f} years")
    ax.axvline(duration.mean(), color=COLORS["dark_gray"], linewidth=1.8, linestyle="--", label=f"Mean {duration.mean():.1f} years")
    ax.set_xlabel("Follow-up Duration (Years)", fontsize=18)
    ax.set_ylabel("Case Reports", fontsize=18)
    ax.set_xticks([0, 5, 10, 20, 30, 40])
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.legend(loc="upper right", fontsize=15, handlelength=2.0, facecolor="white", edgecolor="#D1D5DB", framealpha=0.9)
    ax.yaxis.grid(True, color=COLORS["light_gray"], linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.97)
    save(fig, "case_report_followup_horizon")
    plt.close(fig)


def plot_top_journals(df: pd.DataFrame) -> None:
    all_journal_counts = df["journal"].value_counts().drop(index="Others", errors="ignore")
    selected_journals = all_journal_counts.head(10).index.tolist()
    target_journal = "Clinical Implant Dentistry and Related Research"
    if target_journal in all_journal_counts.index and target_journal not in selected_journals:
        selected_journals.append(target_journal)
    selected_journals = [name for name in selected_journals if name != target_journal]
    journal_counts = all_journal_counts.loc[selected_journals].sort_values()
    if target_journal in all_journal_counts.index:
        journal_counts.loc[target_journal] = int(all_journal_counts[target_journal])
    if "Others" in df["journal"].values:
        others_count = int((df["journal"] == "Others").sum())
        journal_counts.loc["Others"] = others_count
    display_order = [
        name for name in ("Others", target_journal)
        if name in journal_counts.index
    ] + [
        name for name in journal_counts.index
        if name not in {"Others", target_journal}
    ]
    journal_counts = journal_counts.loc[display_order]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    y_positions = list(range(len(journal_counts)))
    top_four = set(journal_counts.nlargest(4).index)
    line_colors = [COLORS["blue_dark"] if i % 2 == 0 else COLORS["pink_dark"] for i in y_positions]
    for y, value, color in zip(y_positions, journal_counts.values, line_colors):
        ax.hlines(y, 0, value, color=color, linewidth=3.2)
    ax.scatter(journal_counts.values, y_positions, color=line_colors, s=110, zorder=3)
    for y, (journal, value) in enumerate(journal_counts.items()):
        ax.text(
            value + 1.8,
            y,
            str(value),
            va="center",
            fontsize=18,
            fontproperties=TIMES_NEW_ROMAN_BOLD,
            color=COLORS["dark_gray"],
        )
        if journal in top_four:
            ax.text(
                value / 2,
                y + 0.22,
                journal,
                ha="center",
                va="bottom",
                fontsize=15,
                fontproperties=TIMES_NEW_ROMAN_ITALIC,
                color=COLORS["dark_gray"],
            )
        else:
            ax.text(
                value + 6.0,
                y,
                journal,
                ha="left",
                va="center",
                fontsize=15,
                fontproperties=TIMES_NEW_ROMAN_ITALIC,
                color=COLORS["dark_gray"],
            )
    ax.set_yticks([])
    ax.set_xlabel("Case Reports", fontsize=18)
    ax.set_xlim(0, 80)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
    ax.xaxis.grid(True, color=COLORS["light_gray"], linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.17, top=0.97)
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
    plot_top_journals(df)
    write_summary(df)


if __name__ == "__main__":
    main()
