"""Bicep curl definition. Best viewed from the side."""
from exercises.base_exercise import Exercise, FormCheck

BICEP_CURL = Exercise(
    key="bicep_curl",
    name="Bicep Curl",
    rep_feature="elbow_angle",
    start_extended=True,
    phase_names=("ARM EXTENDED", "CURLING", "TOP", "LOWERING"),
    checks=(
        FormCheck("curl_height", "Curl height", "elbow_angle", "min",
                  "Curl higher: elbow angle only reached {value:.0f} deg", weight=1.5),
        FormCheck("extension", "Full extension", "elbow_angle", "max",
                  "Extend the arm fully between reps: elbow only opened to {value:.0f} deg"),
        FormCheck("elbow_stability", "Elbow stability", "upper_arm_elevation", "max",
                  "Keep the upper arm still: it swung {value:.0f} deg away from the torso",
                  live=True),
        FormCheck("torso_swing", "Torso stability", "torso_inclination", "range",
                  "Avoid swinging the torso: it moved {value:.0f} deg during the rep"),
    ),
    display_features=("elbow_angle", "shoulder_angle"),
    description="Elbow flexion with a still upper arm; rep = extended -> curled -> extended.",
)
