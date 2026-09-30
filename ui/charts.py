"""Matplotlib charts for the Results page and the PDF report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")      # headless backend: works inside Streamlit and tests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.scoring import status
from core.session import SessionLog
from exercises import EXERCISES

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
SERIES = "#2a78d6"
STATUS_HEX = {"good": "#0ca30c", "warning": "#fab219", "bad": "#d03b3b", "neutral": MUTED}


def _figure(width: float = 8, height: float = 3.2) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _labels(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_2)


def angle_vs_time(session: SessionLog, exercise_key: str) -> Figure:
    """Rep-driving joint angle over time, with counted reps shaded and thresholds."""
    exercise = EXERCISES[exercise_key]
    feature = exercise.rep_feature
    df = session.frames_df()
    fig, ax = _figure()
    if feature in df:
        ax.plot(df["t"], df[feature], color=SERIES, linewidth=2)
    for r in session.reps:
        if r.exercise == exercise_key:
            ax.axvspan(r.start_time, r.end_time, color=SERIES, alpha=0.08, linewidth=0)
            ax.text((r.start_time + r.end_time) / 2, 0.97, f"#{r.index}", fontsize=7,
                    color=INK_2, ha="center", va="top", transform=ax.get_xaxis_transform())
    thr = exercise.rep_thresholds
    for name in ("extended", "flexed"):
        ax.axhline(thr[name], color=MUTED, linewidth=1, linestyle="--")
        ax.text(ax.get_xlim()[1], thr[name], f" {name} {thr[name]}", fontsize=7,
                color=MUTED, va="center")
    _labels(ax, f"{feature.replace('_', ' ').title()} vs time (shaded = counted reps)",
            "time (s)", "angle (deg)")
    fig.tight_layout()
    return fig


def score_per_rep(session: SessionLog) -> Figure:
    """Form score of every rep, coloured by status band and labelled."""
    fig, ax = _figure()
    reps = [r for r in session.reps if r.score is not None]
    xs = list(range(1, len(reps) + 1))
    ax.bar(xs, [r.score for r in reps], width=0.6,
           color=[STATUS_HEX[status(r.score)] for r in reps])
    for x, r in zip(xs, reps):
        ax.text(x, r.score + 1.5, f"{r.score:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 110)
    multi = len({r.exercise for r in reps}) > 1
    ax.set_xticks(xs, [f"{EXERCISES[r.exercise].name}\n#{r.index}" if multi else f"#{r.index}"
                       for r in reps])
    _labels(ax, "Form score per repetition (green >= 75, yellow >= 50, red < 50)",
            "repetition", "score (0-100)")
    fig.tight_layout()
    return fig


def rep_durations(session: SessionLog) -> Figure:
    fig, ax = _figure()
    xs = list(range(1, len(session.reps) + 1))
    ax.bar(xs, [r.duration for r in session.reps], width=0.6, color=SERIES)
    for x, r in zip(xs, session.reps):
        ax.text(x, r.duration + 0.03, f"{r.duration:.1f}s", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, [f"#{r.index}" for r in session.reps])
    _labels(ax, "Repetition duration (movement phase)", "repetition", "seconds")
    fig.tight_layout()
    return fig


def check_breakdown(session: SessionLog, exercise_key: str) -> Figure:
    """Average score of every form check (horizontal bars)."""
    df = session.check_breakdown(exercise_key).dropna(subset=["average_score"])
    fig, ax = _figure(height=0.45 * max(len(df), 2) + 0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ys = list(range(len(df)))[::-1]
    ax.barh(ys, df["average_score"], height=0.55,
            color=[STATUS_HEX[status(s)] for s in df["average_score"]])
    for y, s in zip(ys, df["average_score"]):
        ax.text(s + 1, y, f"{s:.0f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, df["check"])
    ax.set_xlim(0, 110)
    _labels(ax, f"{EXERCISES[exercise_key].name}: score breakdown", "average score", "")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive (Altair) versions for the dark UI; the matplotlib figures above
# stay as they are for the PDF report.
# ---------------------------------------------------------------------------
import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402

from core.scoring import grade  # noqa: E402

UI_INK, UI_MUTED, UI_LINE, UI_ACCENT = "#EDECE8", "#85837D", "#242427", "#F4513C"
BANDS = {"good": "Good (75+)", "warning": "Check (50-74)", "bad": "Poor (<50)"}
BAND_COLORS = ["#55C08A", "#E7B24A", "#F4513C"]
MAX_POINTS = 2500


def _ui(chart: alt.Chart, height: int) -> alt.Chart:
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(gridColor=UI_LINE, domainColor="#34343A", tickColor="#34343A",
                            labelColor=UI_MUTED, titleColor=UI_MUTED, labelFont="IBM Plex Mono",
                            titleFont="IBM Plex Sans", labelFontSize=11, titleFontSize=11,
                            titleFontWeight=500, labelPadding=6)
            .configure_legend(labelColor="#B4B2AC", labelFont="IBM Plex Sans", labelFontSize=12,
                              symbolType="square", orient="top", direction="horizontal", title=None))


def ui_score_per_rep(session: SessionLog) -> alt.Chart:
    reps = [r for r in session.reps if r.score is not None]
    multi = len({r.exercise for r in reps}) > 1
    df = pd.DataFrame([{
        "rep": f"{EXERCISES[r.exercise].name} #{r.index}" if multi else f"#{r.index}", "order": i,
        "score": round(r.score, 1), "band": BANDS[status(r.score)], "grade": grade(r.score),
        "duration (s)": round(r.duration, 2), "issues": "; ".join(r.issues) or "none"}
        for i, r in enumerate(reps)])
    x = alt.X("rep:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=0, grid=False))
    base = alt.Chart(df).encode(x=x)
    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, width=alt.RelativeBandSize(0.55)).encode(
        y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100]), title="form score"),
        color=alt.Color("band:N", scale=alt.Scale(domain=list(BANDS.values()), range=BAND_COLORS)),
        tooltip=["rep", "score", "grade", "duration (s)", "issues"])
    labels = base.mark_text(dy=-9, font="IBM Plex Mono", fontSize=11, color=UI_INK).encode(
        y="score:Q", text=alt.Text("score:Q", format=".0f"))
    rules = alt.Chart(pd.DataFrame({"y": [75, 50]})).mark_rule(
        strokeDash=[3, 4], color=UI_MUTED, opacity=0.7).encode(y="y:Q")
    layers = rules + bars
    if len(df) <= 16:          # direct labels only while they stay legible
        layers += labels
    return _ui(layers, 280)


def ui_rep_durations(session: SessionLog) -> alt.Chart:
    df = pd.DataFrame([{"rep": f"#{r.index}", "order": i, "seconds": round(r.duration, 2)}
                       for i, r in enumerate(session.reps)])
    bars = alt.Chart(df).mark_bar(color="#6E6C67", cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                  width=alt.RelativeBandSize(0.55)).encode(
        x=alt.X("rep:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y("seconds:Q", title="movement time (s)"), tooltip=["rep", "seconds"])
    return _ui(bars, 280)


def ui_angle_vs_time(session: SessionLog, exercise_key: str) -> alt.Chart:
    """Rep-driving joint angle over time; counted reps shaded, thresholds dashed, hover read-out."""
    exercise = EXERCISES[exercise_key]
    feature = exercise.rep_feature
    title = feature.replace("_", " ")
    df = session.frames_df()
    df = (df[["t", feature]].dropna() if feature in df else pd.DataFrame(columns=["t", feature]))
    df = df.iloc[::max(len(df) // MAX_POINTS, 1)].rename(columns={feature: "angle"})
    df["t"], df["angle"] = df["t"].round(2), df["angle"].astype(float).round(1)
    reps = pd.DataFrame([{"start": r.start_time, "end": r.end_time, "mid": (r.start_time + r.end_time) / 2,
                          "label": f"#{r.index}"} for r in session.reps if r.exercise == exercise_key]
                        or [{"start": None, "end": None, "mid": None, "label": ""}])
    thr = exercise.rep_thresholds
    t_max = float(df["t"].max()) if len(df) else 1.0
    lines = pd.DataFrame([{"y": thr[n], "label": f"{n} {thr[n]}", "t": t_max} for n in ("extended", "flexed")])
    x = alt.X("t:Q", title="time (s)", scale=alt.Scale(nice=False), axis=alt.Axis(grid=False))
    shade = alt.Chart(reps).mark_rect(color=UI_ACCENT, opacity=0.1).encode(x="start:Q", x2="end:Q")
    rep_labels = alt.Chart(reps).mark_text(font="IBM Plex Mono", fontSize=10, color=UI_MUTED, dy=-4).encode(
        x="mid:Q", y=alt.value(10), text="label:N")
    rules = alt.Chart(lines).mark_rule(strokeDash=[3, 4], color=UI_MUTED, opacity=0.7).encode(y="y:Q")
    rule_text = alt.Chart(lines).mark_text(align="right", dy=-7, font="IBM Plex Mono", fontSize=10,
                                           color=UI_MUTED).encode(x="t:Q", y="y:Q", text="label:N")
    hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    line = alt.Chart(df).mark_line(color=UI_INK, strokeWidth=1.8).encode(
        x=x, y=alt.Y("angle:Q", title=f"{title} (deg)", scale=alt.Scale(zero=False)))
    points = alt.Chart(df).mark_circle(size=70, color=UI_ACCENT).encode(
        x=x, y="angle:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("t:Q", title="time (s)"), alt.Tooltip("angle:Q", title=f"{title} (deg)")]
    ).add_params(hover)
    cursor = alt.Chart(df).mark_rule(color="#34343A").encode(x="t:Q").transform_filter(hover)
    return _ui(shade + rules + rule_text + rep_labels + cursor + line + points, 300)
