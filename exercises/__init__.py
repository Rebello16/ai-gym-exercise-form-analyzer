"""Exercise registry. Register new exercises here."""
from exercises.base_exercise import Exercise
from exercises.bicep_curl import BICEP_CURL
from exercises.pushup import PUSHUP
from exercises.shoulder_press import SHOULDER_PRESS
from exercises.squat import SQUAT

EXERCISES: dict[str, Exercise] = {e.key: e for e in (SQUAT, BICEP_CURL, PUSHUP, SHOULDER_PRESS)}
