"""Overhead shoulder press definition. Side view recommended."""
from exercises.base_exercise import Exercise, FormCheck

SHOULDER_PRESS = Exercise(
    key="shoulder_press",
    name="Shoulder Press",
    rep_feature="elbow_angle",
    start_extended=False,
    phase_names=("RACK", "PRESSING", "LOCKOUT", "LOWERING"),
    checks=(
        FormCheck("lockout", "Lockout", "elbow_angle", "max",
                  "Press fully overhead: elbow only extended to {value:.0f} deg", weight=1.5),
        FormCheck("bottom_range", "Range (bottom)", "elbow_angle", "min",
                  "Lower the weight to shoulder level: elbow only bent to {value:.0f} deg"),
        FormCheck("back_arch", "Upright torso", "torso_inclination", "max",
                  "Avoid leaning back: torso tilted {value:.0f} deg from vertical", live=True),
        FormCheck("wrist_stack", "Wrist over elbow", "forearm_from_vertical", "max",
                  "Keep wrists stacked over elbows: forearm tilted {value:.0f} deg", live=True),
    ),
    display_features=("elbow_angle", "shoulder_angle"),
    description="Vertical pressing; rep = rack (hands at shoulders) -> lockout -> rack.",
)
