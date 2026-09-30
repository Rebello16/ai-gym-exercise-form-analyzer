"""Squat definition. Best viewed from the side (depth/lean) or front (knee alignment)."""
from exercises.base_exercise import Exercise, FormCheck

SQUAT = Exercise(
    key="squat",
    name="Squat",
    rep_feature="knee_angle",
    start_extended=True,
    phase_names=("STANDING", "DESCENDING", "BOTTOM", "ASCENDING"),
    checks=(
        FormCheck("depth", "Depth", "knee_angle", "min",
                  "Go deeper: knee angle only reached {value:.0f} deg", weight=1.5),
        FormCheck("back_angle", "Back angle", "torso_inclination", "max",
                  "Keep your back more upright: torso leaned {value:.0f} deg from vertical",
                  live=True),
        FormCheck("knee_tracking", "Knee tracking", "knee_over_toe", "max",
                  "Knees travelled far past the toes ({value:.2f} shin lengths)", live=True),
        FormCheck("knee_alignment", "Knee alignment", "knee_width_ratio", "min",
                  "Keep your knees aligned with your feet: knees moved inward "
                  "(knee/ankle width {value:.2f})", live=True),
    ),
    display_features=("knee_angle", "hip_angle"),
    description="Knee-driven lower-body movement; rep = stand -> bottom -> stand.",
)
