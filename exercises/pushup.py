"""Push-up definition. Must be viewed from the side."""
from exercises.base_exercise import Exercise, FormCheck

PUSHUP = Exercise(
    key="pushup",
    name="Push-Up",
    rep_feature="elbow_angle",
    start_extended=True,
    phase_names=("TOP", "LOWERING", "BOTTOM", "PUSHING UP"),
    checks=(
        FormCheck("depth", "Depth", "elbow_angle", "min",
                  "Lower your chest further: elbow angle only reached {value:.0f} deg",
                  weight=1.5),
        FormCheck("lockout", "Lockout", "elbow_angle", "max",
                  "Extend your arms fully at the top: elbow only opened to {value:.0f} deg"),
        FormCheck("hip_sag", "Body alignment (sag)", "hip_line_offset", "max",
                  "Hips are sagging below the shoulder-ankle line ({value:.2f} body lengths)",
                  weight=1.25, live=True),
        FormCheck("hip_pike", "Body alignment (pike)", "hip_line_offset", "min",
                  "Hips are piked above the shoulder-ankle line ({value:.2f} body lengths)",
                  live=True),
    ),
    display_features=("elbow_angle", "hip_angle"),
    description="Horizontal pressing; rep = arms extended -> chest low -> arms extended.",
)
