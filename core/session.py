"""Session log + summary statistics (pure data, no UI)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from config.thresholds import STATUS_GOOD
from core.feature_extractor import FEATURE_UNITS
from core.form_analyzer import RepResult
from core.scoring import grade
from exercises import EXERCISES


@dataclass
class SessionLog:
    """Everything recorded during one analysis session."""

    source: str
    mode: str
    started_at: datetime = field(default_factory=datetime.now)
    frames: list[dict] = field(default_factory=list)
    reps: list[RepResult] = field(default_factory=list)
    partials: list[RepResult] = field(default_factory=list)

    # -- derived ----------------------------------------------------------
    @property
    def duration(self) -> float:
        return self.frames[-1]["t"] - self.frames[0]["t"] if len(self.frames) > 1 else 0.0

    def frames_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.frames)

    def reps_df(self) -> pd.DataFrame:
        rows = []
        for r in self.reps:
            row = {"rep": r.index, "exercise": EXERCISES[r.exercise].name,
                   "start_s": round(r.start_time, 2), "end_s": round(r.end_time, 2),
                   "duration_s": round(r.duration, 2),
                   "score": None if r.score is None else round(r.score, 1),
                   "issues": " | ".join(r.issues)}
            for key, value in r.scores.items():
                row[f"score_{key}"] = None if value is None else round(value, 1)
            for key, value in r.values.items():
                row[f"value_{key}"] = None if value is None else round(value, 3)
            rows.append(row)
        return pd.DataFrame(rows)

    def exercises_done(self) -> list[str]:
        return sorted({r.exercise for r in self.reps})

    def check_breakdown(self, exercise_key: str) -> pd.DataFrame:
        """Average score of every check across the reps of one exercise."""
        exercise = EXERCISES[exercise_key]
        reps = [r for r in self.reps if r.exercise == exercise_key]
        rows = []
        for check in exercise.all_checks:
            scores = [r.scores[check.key] for r in reps if r.scores.get(check.key) is not None]
            rows.append({"check": check.label,
                         "average_score": round(float(np.mean(scores)), 1) if scores else None,
                         "reps_measured": len(scores)})
        return pd.DataFrame(rows)

    def angle_statistics(self) -> pd.DataFrame:
        """min / max / mean / std of every feature over frames with a pose."""
        df = self.frames_df()
        rows = []
        for name, unit in FEATURE_UNITS.items():
            if name not in df or df[name].dropna().empty:
                continue
            s = df[name].dropna()
            rows.append({"feature": name, "unit": unit, "min": round(s.min(), 2),
                         "max": round(s.max(), 2), "mean": round(s.mean(), 2),
                         "std": round(s.std(ddof=0), 2), "samples": int(s.size)})
        return pd.DataFrame(rows)

    def issue_counts(self) -> pd.DataFrame:
        """How many reps triggered each check's feedback."""
        counts: dict[tuple[str, str], int] = {}
        for r in self.reps:
            exercise = EXERCISES[r.exercise]
            for check in exercise.all_checks:
                s = r.scores.get(check.key)
                if s is not None and s < STATUS_GOOD:
                    key = (exercise.name, check.label)
                    counts[key] = counts.get(key, 0) + 1
        return pd.DataFrame([{"exercise": e, "issue": c, "reps_affected": n}
                             for (e, c), n in sorted(counts.items(), key=lambda kv: -kv[1])])

    def summary(self) -> dict:
        scored = [r for r in self.reps if r.score is not None]
        avg = float(np.mean([r.score for r in scored])) if scored else None
        best = max(scored, key=lambda r: r.score) if scored else None
        worst = min(scored, key=lambda r: r.score) if scored else None
        detected = sum(1 for f in self.frames if f.get("pose_detected"))
        return {
            "date_time": self.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.source,
            "mode": self.mode,
            "exercises": ", ".join(EXERCISES[k].name for k in self.exercises_done()) or "-",
            "total_reps": len(self.reps),
            "partial_reps": len(self.partials),
            "average_score": None if avg is None else round(avg, 1),
            "grade": grade(avg),
            "best_rep": None if best is None else f"#{best.index} ({best.score:.0f})",
            "worst_rep": None if worst is None else f"#{worst.index} ({worst.score:.0f})",
            "duration_s": round(self.duration, 1),
            "frames_analysed": len(self.frames),
            "pose_detection_rate": round(detected / len(self.frames), 3) if self.frames else 0.0,
        }
