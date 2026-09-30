"""Report export: CSV (always) and PDF (via matplotlib, no extra dependency)."""
from __future__ import annotations

import io
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from core.session import SessionLog

DISCLAIMER = ("Scores describe geometric deviations detected by computer vision; "
              "they are not medical or professional fitness advice.")


def report_csv(session: SessionLog) -> str:
    """One long-format CSV: section, item, metric, value.

    Contains the summary, per-rep results, detected issues and joint-angle
    statistics, so it opens cleanly in Excel and is still machine-readable.
    """
    rows = [("summary", "", k, v) for k, v in session.summary().items()]
    for _, rep in session.reps_df().iterrows():
        for col, val in rep.items():
            if col != "rep":
                rows.append(("rep", f"{rep['exercise']} #{rep['rep']}", col, val))
    for _, row in session.issue_counts().iterrows():
        rows.append(("form_issue", f"{row['exercise']}: {row['issue']}", "reps_affected",
                     row["reps_affected"]))
    for _, row in session.angle_statistics().iterrows():
        for metric in ("min", "max", "mean", "std", "samples"):
            rows.append(("angle_statistics", f"{row['feature']} ({row['unit']})", metric, row[metric]))
    rows.append(("note", "", "disclaimer", DISCLAIMER))
    return pd.DataFrame(rows, columns=["section", "item", "metric", "value"]).to_csv(index=False)


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
    fig.text(0.07, 0.95, title, fontsize=16, fontweight="bold")
    y = 0.91
    for line in lines:
        for wrapped in textwrap.wrap(line, 95) or [""]:
            fig.text(0.07, y, wrapped, fontsize=9, family="monospace")
            y -= 0.018
            if y < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def report_pdf(session: SessionLog) -> bytes:
    """Multi-page PDF: summary, reps, issues, angle stats, then charts."""
    from ui import charts   # local import keeps CSV export free of plotting

    summary = session.summary()
    lines = [f"{k.replace('_', ' ').title():<24}: {v}" for k, v in summary.items()]
    lines += ["", DISCLAIMER, "", "REPETITIONS"]
    reps = session.reps_df()
    if reps.empty:
        lines.append("  No repetitions were counted.")
    for _, r in reps.iterrows():
        lines.append(f"  {r['exercise']} #{r['rep']}: score {r['score']}, {r['duration_s']} s")
        if r["issues"]:
            lines += [f"      - {m}" for m in r["issues"].split(" | ")]
    lines += ["", "FORM ISSUES (reps affected)"]
    for _, r in session.issue_counts().iterrows():
        lines.append(f"  {r['exercise']}: {r['issue']} - {r['reps_affected']}")
    lines += ["", "JOINT-ANGLE STATISTICS"]
    stats = session.angle_statistics()
    if not stats.empty:
        lines += stats.to_string(index=False).splitlines()

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        _text_page(pdf, "Exercise Form Analysis Report", lines)
        figures = [charts.score_per_rep(session), charts.rep_durations(session)]
        for key in session.exercises_done():
            figures += [charts.angle_vs_time(session, key), charts.check_breakdown(session, key)]
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    return buffer.getvalue()
