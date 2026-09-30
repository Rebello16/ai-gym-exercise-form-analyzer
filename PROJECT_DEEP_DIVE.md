# AI-Based Gym Exercise Form Analyzer: Project Deep Dive

This document explains how the project **actually works**, file by file and function by function,
so that you can explain every part of it in a viva or presentation. Everything here was written
after reading the current source code and running it. Where the code differs from what a reader
might assume, the document says what the code does.

**How to use this document**

* Every implementation detail is followed by a code reference in the form
  `file.py → function()` or `file.py → Class.method()`. Search the repository for that name to
  find the code.
* Numbers in examples (angles, scores, timings) come from real runs of this code on the demo clip
  `squat_bad`, unless the text says the example is illustrative.
* The phrase **"not implemented"** is used explicitly wherever a common technique is *not* part
  of this project, so you never claim something that is not in the code.

**Contents**

1. Project overview
2. One-sentence explanation
3. Complete system flow
4. Complete project structure
5. File-by-file deep explanation
6. Application startup
7. Streamlit UI (every page)
8. Navigation
9. Image input
10. Image preprocessing
11. Image processing pipeline (what exists and what does not)
12. Pose estimation
13. Pose landmark diagram
14. Mathematics
15. Normalization
16. Feature extraction
17. Exercise recognition
18. Squat analysis
19. Bicep curl analysis
20. Push-up analysis
21. Shoulder press analysis
22. Repetition counting
23. Form analysis
24. Scoring system
25. Visualization
26. Video processing
27. Demo mode
28. Session state
29. Data flow table
30. Error handling
31. Performance
32. Smoothing and noise reduction
33. Complete user example: one squat, frame by frame
34. What happens when you click each button
35. DIP concepts mapped to the project
36. Why each technology was chosen
37. Alternatives
38. Limitations
39. Future improvements
40. Common viva questions (50 questions)
41. 30-second explanation
42. 1-minute explanation
43. 3-minute explanation
44. Presentation walkthrough
45. Code references index
46. Code snippets, explained line by line
47. Glossary
48. If you remember only 10 things
49. The entire project in one diagram

---

# 1. Project overview

## 1.1 In simple language

When people train alone at the gym, nobody tells them whether their squat is deep enough, whether
their back is leaning too far, or whether they are swinging their arms during a curl. This project
turns an ordinary webcam (or a recorded video) into a simple "digital coach".

You stand in front of the camera and exercise. The program:

1. looks at every video frame,
2. cleans up the image if the lighting is bad or the picture is noisy,
3. finds your body and marks 33 points on it (shoulders, elbows, hips, knees, ankles and so on),
4. measures the angles of your joints from those points,
5. works out which exercise you are doing (squat, bicep curl, push-up or shoulder press),
6. counts your repetitions,
7. checks each repetition against rules (for example "the knee should bend to about 95 degrees
   at the bottom of a squat"),
8. gives each repetition a score out of 100 and tells you in plain words what to fix,
9. draws all of this over the video in real time, and
10. produces a report with charts that you can download as CSV or PDF.

It runs on an ordinary laptop CPU. No GPU, no wearable sensors, no internet connection (except to
download the pose model once).

## 1.2 Why it was built

The project is a Digital Image Processing (DIP) project. It shows DIP techniques (resizing,
colour-space conversion, histogram statistics, noise estimation, Gaussian filtering, CLAHE,
Laplacian edge detection, thresholding, morphology, region of interest) working together with a
computer-vision model (pose estimation) to solve a practical problem.

## 1.3 The problem it solves

Given a video of **one person** exercising, automatically determine:

* **which** exercise is being performed,
* **how many** repetitions were completed (and which attempts were only partial),
* **how good** the form was in each repetition,
* **why** a repetition lost points (which measurable deviation caused it),

in real time, on commodity hardware.

## 1.4 Main features (as implemented)

| Feature | Where it lives |
|---|---|
| Live webcam analysis with Start / Stop and a session time limit | `ui/pages.py → live()` |
| Video upload (MP4 / AVI / MOV) with live preview and an annotated output video | `ui/pages.py → upload()`, `processing/video_processor.py → analyze_video()` |
| Demo mode with 10 scripted, rendered clips and batch validation | `ui/pages.py → demo()`, `demo/synthetic_athlete.py` |
| Adaptive preprocessing (resize, CLAHE, Gaussian denoise only when measured as needed) | `processing/preprocessing.py → preprocess_frame()` |
| Pose detection, 33 landmarks, multi-person handling | `core/pose_detector.py → PoseDetector.detect()` |
| Joint angles and scale-invariant geometric features | `core/feature_extractor.py → FeatureExtractor.extract()` |
| Automatic (rule-based) exercise recognition, or manual selection | `core/exercise_recognizer.py → ExerciseRecognizer` |
| Repetition counting with a hysteresis state machine | `core/rep_counter.py → RepCounter.update()` |
| Per-rep form checks, per-frame posture checks, feedback messages | `core/form_analyzer.py → FormAnalyzer` |
| Transparent 0-100 scoring with weights and grade bands | `core/scoring.py` |
| Skeleton, angle and feedback overlay on the video | `processing/visualization.py → draw_overlay()` |
| Session report: headline numbers, form breakdown, charts, tables | `ui/pages.py → results()`, `ui/charts.py` |
| CSV and PDF export | `utils/export.py → report_csv()`, `report_pdf()` |
| DIP pipeline viewer (every processing stage on one frame) | `ui/components.py → dip_stage_viewer()` |
| Skeleton illustrations generated from real pose data | `ui/figures.py` |

## 1.5 Technologies used

| Technology | What this project uses it for |
|---|---|
| **Python 3.11+** (developed on 3.14) | The whole application. |
| **OpenCV** (`cv2`, installed as `opencv-contrib-python`) | Reading webcam and video frames, resizing, colour conversion, Gaussian blur, CLAHE, Laplacian, morphology, drawing the skeleton, writing the annotated video. |
| **MediaPipe Tasks** (`mediapipe`) | The Pose Landmarker model (BlazePose, *lite* variant) that finds 33 body landmarks. |
| **NumPy** | Landmark arrays, percentiles and medians in recognition, the noise-estimation sum, mask arithmetic. |
| **pandas** | Session tables (per frame, per rep, statistics) and CSV export. |
| **Matplotlib** | Charts inside the PDF report and the histogram in the DIP viewer. |
| **Altair** (installed with Streamlit) | Interactive charts on the Results page. |
| **Streamlit** | The web user interface. |
| **pytest** | 86 automated tests. |

## 1.6 Overall architecture (technical)

The project is a **modular, frame-by-frame pipeline** behind a Streamlit UI.

* One object, `AnalysisPipeline` (`core/pipeline.py`), owns one analysis session. For each frame
  its `process()` method runs: preprocessing, pose detection, ROI and sharpness, feature
  extraction, exercise recognition, the repetition state machines, form analysis, scoring,
  visualisation and logging, in that order.
* The **input side** (`processing/video_processor.py`) produces frames with timestamps, either
  from a file (`iterate_video()`) or from a webcam (`open_webcam()` plus a loop in
  `ui/pages.py → live()`).
* The **exercise definitions** (`exercises/*.py`) are *data*, not code: each exercise is a
  declarative `Exercise` object listing its rep signal, phases and `FormCheck` rules. All numeric
  thresholds live in one file, `config/thresholds.py`.
* The **output side** is a `SessionLog` (`core/session.py`) holding every frame's measurements and
  every rep's result. The Results page, charts and exports read only from it.
* The **UI** (`ui/`) never computes analysis results itself. It creates a pipeline, feeds it
  frames, and displays the `FrameStatus` objects and the `SessionLog` that the pipeline returns.

The whole system is **rule-based and transparent**. The only learned (machine-learning) component
is the MediaPipe pose model, which is used as a pre-trained black box to obtain landmarks.
Exercise recognition, rep counting, form analysis and scoring are hand-written geometric rules
with thresholds you can read in `config/thresholds.py`.

---

# 2. One-sentence explanation

> "Our system takes webcam or video frames, adaptively enhances them with image-processing
> techniques like CLAHE and Gaussian filtering, uses MediaPipe pose estimation to find 33 body
> landmarks, turns those landmarks into joint angles and body-proportion features, recognises the
> exercise with geometric rules, counts repetitions with a hysteresis state machine, scores every
> repetition against measurable form rules, and shows the feedback live on the video and in a
> downloadable report."

Shorter version:

> "It's a camera-based gym coach: image processing plus pose estimation give us joint angles,
> and transparent rules turn those angles into rep counts, form scores and feedback."

---

# 3. Complete system flow

## 3.1 From opening the app to the final result

```text
User runs:  streamlit run app.py
 │
 ▼
app.py  (runs top to bottom on every interaction)
 ├─ st.set_page_config, inject CSS                        app.py
 ├─ decide the page (query param / session state)         app.py
 ├─ read_settings()  -> Settings object                   ui/components.py
 ├─ top_nav()        -> top navigation bar                ui/pages.py
 └─ PAGE_FUNCS[page](settings) -> draws one page          ui/pages.py
        │
        │ user presses Start (camera) / Analyse video / Analyse this clip
        ▼
make_pipeline(settings, source)                            ui/components.py
 ├─ PoseDetector(...)     (loads the MediaPipe model)      core/pose_detector.py
 └─ AnalysisPipeline(...) (fresh state for this session)   core/pipeline.py
        │
        ▼
FRAME SOURCE
 ├─ camera: open_webcam() + cap.read() loop                processing/video_processor.py, ui/pages.py → live()
 └─ file:   iterate_video(path, frame_skip)                processing/video_processor.py
        │   (frame_bgr, t seconds)
        ▼
AnalysisPipeline.process(frame_bgr, t)                     core/pipeline.py
 │
 ├─ 1 PREPROCESSING      preprocess_frame()                processing/preprocessing.py
 │     resize (INTER_AREA) → grey → brightness/contrast on ROI → noise σ
 │     → [Gaussian denoise if noisy] → [CLAHE if dark/flat] → BGR→RGB
 │
 ├─ 2 POSE DETECTION     PoseDetector.detect(rgb, ms)      core/pose_detector.py
 │     MediaPipe PoseLandmarker (VIDEO mode) → up to 2 people
 │     → most prominent person → 33 (x, y) in pixels + visibility
 │
 ├─ 3 SUBJECT PROCESSING landmarks_roi(), sharpness()      processing/image_processing.py
 │     padded bounding box (ROI) → variance of Laplacian
 │     [binary_person_mask() + suppress_background() if "Background focus" is on]
 │
 ├─ 4 FEATURES           FeatureExtractor.extract()        core/feature_extractor.py
 │     body side + view → 15 features (angles, ratios) → EMA smoothing
 │
 ├─ 5 RECOGNITION        ExerciseRecognizer.update()       core/exercise_recognizer.py
 │     2-second window → 4 fuzzy scores → best ≥ 0.5 → majority vote
 │     (skipped as the decision if the user forced an exercise)
 │
 ├─ 6 REP COUNTING       ExerciseTracker.update()          core/pipeline.py
 │     → RepCounter.update()                               core/rep_counter.py
 │     one hysteresis state machine per exercise, all running in parallel
 │
 ├─ 7 FORM ANALYSIS      FormAnalyzer.evaluate_rep() / live()   core/form_analyzer.py
 │     per rep: median filter → min/max/range → check scores → issues
 │     per frame: "live" posture checks
 │
 ├─ 8 SCORING            check_score(), weighted_score()   core/scoring.py
 │
 ├─ 9 FEEDBACK/WARNINGS  _compose_feedback(), _collect_warnings()   core/pipeline.py
 │
 ├─10 VISUALISATION      draw_overlay()                    processing/visualization.py
 │     skeleton + angle labels + info panel + messages + status border
 │
 └─11 LOGGING            _log() → SessionLog.frames        core/pipeline.py, core/session.py
        │
        ▼ returns (annotated_bgr, FrameStatus)
UI shows the annotated frame and status_panel(FrameStatus)  ui/components.py
(file input: annotated frames are also written to outputs/<name>_analyzed.webm)
        │
        ▼ when the video ends / Stop is pressed
st.session_state["last_session"] = pipeline.session
        │
        ▼ user opens Results
results() reads SessionLog → summary(), check_breakdown(), charts, tables, exports
```

## 3.2 Every stage: input, processing, output, file, function

| # | Stage | Input | Processing | Output | File → function |
|---|---|---|---|---|---|
| 0 | Frame acquisition | camera index or file path | `cv2.VideoCapture` reads frames; timestamps from frame index ÷ FPS (file) or wall clock (camera) | `(frame_bgr, t)` | `processing/video_processor.py → iterate_video()`, `open_webcam()`; `ui/pages.py → live()` |
| 1 | Preprocessing | raw BGR frame, previous ROI | resize ≤ 640 px wide, grey statistics on ROI, noise estimate, conditional Gaussian blur and CLAHE, BGR→RGB | `PreprocessResult` (bgr, rgb, scale, brightness, contrast, noise σ, list of applied operations) | `processing/preprocessing.py → preprocess_frame()` |
| 2 | Pose detection | RGB frame, timestamp in ms | MediaPipe Pose Landmarker in VIDEO mode; normalized coordinates × width/height; choose most prominent person | `PoseDetection` (33×2 points in pixels, 33 visibilities, number of people, optional mask, error) | `core/pose_detector.py → PoseDetector.detect()` |
| 3 | Subject processing | landmarks, processed frame | padded bounding box of visible landmarks; variance of Laplacian inside it; optional mask threshold + morphology + background blur | ROI box, sharpness value, optional focused display frame | `processing/image_processing.py → landmarks_roi()`, `sharpness()`, `binary_person_mask()`, `suppress_background()` |
| 4 | Feature extraction | `PoseDetection` | choose body side and view, compute angles/ratios with the geometry helpers, EMA smoothing | `Features` (dict of 15 values, side, view) | `core/feature_extractor.py → FeatureExtractor.extract()` |
| 5 | Exercise recognition | features over the last 2 s | orientation, knee/elbow range of motion, hand height → four fuzzy scores → vote | `Recognition` (label, confidence, scores) | `core/exercise_recognizer.py → ExerciseRecognizer.update()`, `classify()` |
| 6 | Rep counting | the exercise's rep feature (an angle), t | 5-state hysteresis state machine | state (phase) and, when a movement ends, a `RepEvent` ("rep" or "partial") | `core/rep_counter.py → RepCounter.update()` via `core/pipeline.py → ExerciseTracker.update()` |
| 7 | Form analysis | all feature dicts of the finished rep | median filter, min/max/range, check score, feedback message | `RepResult` (values, scores, score, issues) | `core/form_analyzer.py → FormAnalyzer.evaluate_rep()` |
| 7b | Live posture checks | current frame's features | only checks marked `live=True` | list of `LiveCheck` | `core/form_analyzer.py → FormAnalyzer.live()` |
| 8 | Scoring | measured value, (ideal, zero), weights | linear clipped score, weighted mean, grade band, status colour | 0-100 numbers, grade text | `core/scoring.py → check_score()`, `weighted_score()`, `grade()`, `status()` |
| 9 | Feedback and warnings | live checks, last rep, pose, preprocessing | messages and overall status (worst wins) | `FrameStatus.feedback`, `.warnings`, `.status` | `core/pipeline.py → AnalysisPipeline._compose_feedback()`, `_collect_warnings()` |
| 10 | Visualisation | frame + `FrameStatus` | OpenCV lines, circles, text, translucent panel, coloured border | annotated BGR frame | `processing/visualization.py → draw_overlay()` |
| 11 | Logging | `FrameStatus`, features, preprocessing | append one dict per frame | `SessionLog.frames`, `.reps`, `.partials` | `core/pipeline.py → AnalysisPipeline._log()` |
| 12 | Output video | annotated frames | VP8 WebM writer (MP4 fallback) | `outputs/<name>_analyzed.webm` | `processing/video_processor.py → create_writer()`, `analyze_video()` |
| 13 | Report | `SessionLog` | summaries, statistics, charts, CSV/PDF | Results page, downloads | `core/session.py`, `ui/pages.py → results()`, `ui/charts.py`, `utils/export.py` |

---

# 4. Complete project structure

```text
gym-exercise-form-analyzer/
├── app.py                         Streamlit entry point (routing)
├── requirements.txt               dependencies
├── pytest.ini                     test configuration
├── README.md                      project overview
├── PROJECT_DOCUMENTATION.md       formal project report
├── PROJECT_DEEP_DIVE.md           this document
├── .gitignore
├── .streamlit/
│   └── config.toml                localhost-only server, dark theme, fonts
├── config/
│   ├── __init__.py
│   └── thresholds.py              EVERY tunable number
├── core/                          the analysis engine
│   ├── __init__.py
│   ├── angle_calculator.py        geometry: angle, angle from vertical, distance, line offset
│   ├── pose_detector.py           MediaPipe wrapper, landmark names, skeleton, multi-person
│   ├── feature_extractor.py       landmarks → features, side/view choice, EMA
│   ├── exercise_recognizer.py     rule-based recognition + vote smoothing
│   ├── rep_counter.py             hysteresis state machine
│   ├── form_analyzer.py           per-rep checks, median filter, live checks
│   ├── scoring.py                 check score, rep score, grades, status colours
│   ├── pipeline.py                per-frame orchestration, feedback, warnings, logging
│   └── session.py                 session log, summaries, statistics
├── exercises/                     declarative exercise definitions
│   ├── __init__.py                registry EXERCISES
│   ├── base_exercise.py           Exercise, FormCheck, tempo check
│   ├── squat.py
│   ├── bicep_curl.py
│   ├── pushup.py
│   └── shoulder_press.py
├── processing/                    image processing and video I/O
│   ├── __init__.py
│   ├── preprocessing.py           resize, colour, statistics, noise, Gaussian, CLAHE
│   ├── image_processing.py        ROI, Laplacian sharpness, mask + morphology, background
│   ├── visualization.py           skeleton / angles / panel overlay (OpenCV drawing)
│   └── video_processor.py         video + webcam acquisition, annotated video writer
├── demo/
│   ├── __init__.py
│   ├── synthetic_athlete.py       scripted, rendered demo clips (ground truth)
│   └── dip_experiment.py          preprocessing ablation experiment
├── ui/                            Streamlit user interface
│   ├── __init__.py
│   ├── pages.py                   navigation bar and every page
│   ├── components.py              settings, read-out panel, report blocks, DIP viewer
│   ├── figures.py                 skeleton SVG illustrations from real pose data
│   ├── charts.py                  Matplotlib (PDF) and Altair (UI) charts
│   └── styles.py                  CSS design system
├── utils/
│   ├── __init__.py
│   ├── helpers.py                 model download, file-type check, time formatting
│   ├── export.py                  CSV and PDF report
│   └── logger.py                  logging setup
├── tests/                         86 automated tests
│   ├── test_angles.py
│   ├── test_rep_counter.py
│   ├── test_scoring.py
│   ├── test_exercises.py
│   ├── test_preprocessing.py
│   ├── test_pipeline_integration.py
│   └── test_app.py
├── docs/screenshots/              README images
├── models/                        pose_landmarker_lite.task (downloaded on first use; git-ignored)
├── data/sample/                   demo clips .webm (rendered on first use; git-ignored)
└── outputs/                       annotated videos, uploads (git-ignored)
```

## 4.1 Per-file summary

Below, "Used by" lists the files that import the module.

### `app.py`
* **Purpose:** entry point executed by `streamlit run app.py`. Configures the page, injects CSS,
  picks the current page, builds the `Settings`, draws the navigation bar, calls the page function
  and the footer.
* **Used by:** Streamlit itself; `tests/test_app.py` loads it with `AppTest`.
* **Important code:** `PAGE_FUNCS` dictionary (page name → function), the `try/except RuntimeError`
  around the page call.
* **Data received:** Streamlit session state and URL query parameters.
* **Data produced:** the rendered web page.
* **Why it exists:** Streamlit needs one script to run; keeping it tiny makes routing obvious.

### `config/thresholds.py`
* **Purpose:** every number that tunes behaviour: paths, processing width, preprocessing
  thresholds, pose confidences, smoothing, recognition cues, score bands, per-exercise rep and
  check thresholds, overlay colours.
* **Used by:** almost every module (`preprocessing`, `image_processing`, `pose_detector`,
  `feature_extractor`, `exercise_recognizer`, `rep_counter`, `form_analyzer`, `scoring`,
  `session`, `pipeline`, `visualization`, `video_processor`, `base_exercise`, `helpers`, `ui`).
* **Important names:** `PROCESSING`, `PREPROCESSING`, `POSE`, `FEATURES`, `RECOGNITION`,
  `SCORE_BANDS`, `STATUS_GOOD`, `STATUS_WARNING`, `TEMPO`, `EXERCISE_THRESHOLDS`, `COLORS`.
* **Data received:** nothing (constants).
* **Data produced:** constants.
* **Why it exists:** one place to adjust or explain any threshold, with no numbers hidden in code.

### `core/angle_calculator.py`
* **Purpose:** pure geometry on 2-D image points.
* **Used by:** `core/feature_extractor.py`, `ui/figures.py`, tests.
* **Important functions:** `calculate_angle()`, `angle_from_vertical()`, `distance()`,
  `signed_distance_to_line()`.
* **Data received:** points `(x, y)` in pixels.
* **Data produced:** angles in degrees (0-180), distances, signed normalised offsets.
* **Why it exists:** every exercise rule is built from these four operations.

### `core/pose_detector.py`
* **Purpose:** wraps MediaPipe Pose Landmarker; defines the 33 landmark names (`LANDMARK_NAMES`,
  `LM`) and the skeleton connections (`SKELETON`); chooses the main person when several are visible.
* **Used by:** `core/pipeline.py`, `core/feature_extractor.py`, `processing/visualization.py`,
  `ui/components.py`, `ui/pages.py`, `ui/figures.py`, `demo/dip_experiment.py`, tests.
* **Important names:** `PoseDetector` (`__init__`, `detect`, `close`), `PoseDetection`,
  `prominence()`, `LANDMARK_NAMES`, `LM`, `SKELETON`.
* **Data received:** an RGB frame (NumPy array) and a timestamp in milliseconds.
* **Data produced:** `PoseDetection` with a (33, 2) array of pixel coordinates and a (33,) array
  of visibilities for the chosen person, the number of people, an optional segmentation mask, an
  optional error message and all people's bounding boxes.
* **Why it exists:** isolates the only machine-learning dependency behind a small, testable API.

### `core/feature_extractor.py`
* **Purpose:** converts landmarks into named geometric features (angles and scale-invariant
  ratios), keeps the analysed body side stable, detects front vs side view, smooths values.
* **Used by:** `core/pipeline.py`, `processing/visualization.py` (`FEATURE_VERTEX`),
  `core/session.py` (`FEATURE_UNITS`), `ui/figures.py`, tests.
* **Important names:** `FeatureExtractor` (`extract`, `_update_side`, `_smooth`, `reset`),
  `Features`, `FEATURE_VERTEX`, `FEATURE_UNITS`.
* **Data received:** `PoseDetection`.
* **Data produced:** `Features(values: dict[str, float | None], side, view)`.
* **Why it exists:** raw pixel coordinates depend on where you stand and how big you appear;
  angles and ratios do not, so rules can use fixed thresholds.

### `core/exercise_recognizer.py`
* **Purpose:** decides which exercise is being performed from a 2-second window of features.
* **Used by:** `core/pipeline.py`, tests.
* **Important names:** `ExerciseRecognizer` (`classify`, `update`, `reset`), `Recognition`,
  `_ramp()`, `UNKNOWN`.
* **Data received:** time `t` and the current feature dict each frame.
* **Data produced:** `Recognition(label, confidence, scores)`.
* **Why it exists:** allows "auto detection" so the user does not need to choose the exercise.

### `core/rep_counter.py`
* **Purpose:** counts repetitions of one joint-angle signal with a hysteresis state machine.
* **Used by:** `core/pipeline.py → ExerciseTracker`, `ui/figures.py` (indirectly through
  `ExerciseTracker`), tests.
* **Important names:** `RepCounter` (`update`, `_zone`, `_further`, `in_rep`, `reset`),
  `RepState`, `RepEvent`.
* **Data received:** one angle value (or `None`) and a time per frame.
* **Data produced:** the current state and, when a movement finishes, a `RepEvent`
  (`kind="rep"` or `"partial"`, start time, end time, extreme value).
* **Why it exists:** counting peaks naively would double-count jitter; the state machine only
  counts full, ordered movements.

### `core/form_analyzer.py`
* **Purpose:** scores one repetition from the features recorded during it, and evaluates
  instantaneous posture checks for the live display.
* **Used by:** `core/pipeline.py`, `core/session.py` (`RepResult`), tests.
* **Important names:** `FormAnalyzer` (`evaluate_rep`, `live`, `_score`), `RepResult`,
  `LiveCheck`, `median_filter()`, `_REDUCERS`.
* **Data received:** list of per-frame feature dicts for a rep; or one feature dict.
* **Data produced:** `RepResult` (values, check scores, rep score, issue messages); list of
  `LiveCheck`.
* **Why it exists:** separates "what to measure" (exercise definitions) from "how to measure".

### `core/scoring.py`
* **Purpose:** the scoring formulas and the grade/status bands.
* **Used by:** `core/form_analyzer.py`, `core/pipeline.py`, `core/session.py`,
  `processing/visualization.py`, `ui/components.py`, `ui/pages.py`, `ui/charts.py`.
* **Important functions:** `check_score()`, `weighted_score()`, `grade()`, `status()`.
* **Data received:** numbers.
* **Data produced:** 0-100 scores, grade labels, `good/warning/bad/neutral`.
* **Why it exists:** one transparent place that defines what a score means.

### `core/pipeline.py`
* **Purpose:** orchestrates every stage for one frame and keeps per-session state.
* **Used by:** `processing/video_processor.py`, `ui/components.py`, `ui/pages.py`,
  `ui/figures.py` (`ExerciseTracker`), `demo/`, tests.
* **Important names:** `AnalysisPipeline` (`process`, `_compose_feedback`, `_collect_warnings`,
  `_log`, `rep_count`, `mode`), `ExerciseTracker` (`update`, `phase`), `FrameStatus`.
* **Data received:** a BGR frame and a time.
* **Data produced:** an annotated BGR frame and a `FrameStatus`; appends to `SessionLog`.
* **Why it exists:** the single place where the stages are connected, so the UI and the tests
  drive the exact same code.

### `core/session.py`
* **Purpose:** the record of a session and all derived statistics.
* **Used by:** `core/pipeline.py`, `ui/`, `utils/export.py`.
* **Important names:** `SessionLog` (`frames_df`, `reps_df`, `exercises_done`,
  `check_breakdown`, `angle_statistics`, `issue_counts`, `summary`, `duration`).
* **Data received:** frame dicts and `RepResult`s appended by the pipeline.
* **Data produced:** pandas DataFrames and a summary dict.
* **Why it exists:** the Results page and the exports are pure functions of this log.

### `exercises/base_exercise.py` and `exercises/*.py`
* **Purpose:** declarative definitions. `base_exercise.py` defines the `FormCheck` and `Exercise`
  dataclasses and the shared `TEMPO_CHECK`; each other file builds one `Exercise`.
* **Used by:** `exercises/__init__.py` (registry `EXERCISES`), which is used by the pipeline,
  session, visualisation, UI and tests.
* **Important names:** `Exercise` (`all_checks`, `rep_thresholds`, `check_range`), `FormCheck`,
  `TEMPO_CHECK`, `DURATION`, `SQUAT`, `BICEP_CURL`, `PUSHUP`, `SHOULDER_PRESS`, `EXERCISES`.
* **Why it exists:** adding an exercise needs a new definition file plus thresholds, not new
  pipeline code.

### `processing/preprocessing.py`
* **Purpose:** stage 1, adaptive image preprocessing.
* **Used by:** `core/pipeline.py`, `ui/components.py` (DIP viewer), `demo/dip_experiment.py`, tests.
* **Important names:** `preprocess_frame()`, `resize_frame()`, `grey_statistics()`,
  `estimate_noise_sigma()`, `apply_clahe()`, `PreprocessResult`.
* **Data received:** raw BGR frame, optional ROI from the previous frame.
* **Data produced:** `PreprocessResult`.

### `processing/image_processing.py`
* **Purpose:** stage 3, image operations on the detected subject.
* **Used by:** `core/pipeline.py`, `ui/components.py`, `ui/figures.py`, tests.
* **Important functions:** `landmarks_roi()`, `laplacian_edges()`, `sharpness()`,
  `binary_person_mask()`, `suppress_background()`, `edge_map_for_display()`.

### `processing/visualization.py`
* **Purpose:** draws the analysis onto the frame with OpenCV.
* **Used by:** `core/pipeline.py` (inside `process()`), `ui/components.py` (DIP viewer).
* **Important functions:** `draw_overlay()`, `draw_skeleton()`, `draw_angles()`, `_text()`.

### `processing/video_processor.py`
* **Purpose:** frame acquisition from files and webcams, and the annotated-video writer.
* **Used by:** `ui/components.py`, `ui/pages.py`, `demo/dip_experiment.py`, tests.
* **Important names:** `video_info()`, `iterate_video()`, `create_writer()`, `analyze_video()`,
  `open_webcam()`, `VideoSourceError`.

### `demo/synthetic_athlete.py`
* **Purpose:** renders scripted demo videos of an articulated figure (forward kinematics) so the
  app can be demonstrated and tested without a camera, with known ground truth.
* **Used by:** `ui/pages.py → demo()`, `ui/figures.py`, `demo/dip_experiment.py`, tests.
* **Important names:** `Scenario`, `SCENARIOS`, `SCENARIO_BY_NAME`, `POSES`, `squat_pose()`,
  `curl_pose()`, `press_pose()`, `pushup_pose()`, `draw_figure()`, `render_scenario()`,
  `ensure_samples()`.

### `demo/dip_experiment.py`
* **Purpose:** an ablation experiment measuring how much the preprocessing improves pose detection
  in dark and noisy frames. Run with `python -m demo.dip_experiment`.
* **Important functions:** `degrade()`, `detection_rate()`, `main()`.

### `ui/pages.py`
* **Purpose:** the navigation bar and every page (`home`, `live`, `upload`, `demo`,
  `exercise_guide`, `results`, `about`), plus `footer`.
* **Used by:** `app.py`, tests.

### `ui/components.py`
* **Purpose:** reusable UI parts: `Settings`, `read_settings()`, `settings_bar()`,
  `make_pipeline()`, `status_panel()`, `idle_panel()`, `run_video_analysis()`, `kpi_strip()`,
  `session_overview()`, `read_frame_at()`, `dip_stage_viewer()`, `md()`, `css()`.

### `ui/figures.py`
* **Purpose:** generates the skeleton illustrations (SVG) from the scripted demo athlete, the
  real `SKELETON` topology, real angles (`calculate_angle`) and real phases (the actual
  `ExerciseTracker` state machine). Nothing is hand-drawn.
* **Important functions:** `rep_cycle()`, `joint_angle()`, `rep_vertex()`, `turn_index()`,
  `skeleton()`, `hero_svg()`, `exercise_svg()`, `pose_pair_svg()`, `studies_background()`,
  `camera_placeholder_svg()`.

### `ui/charts.py`
* **Purpose:** Matplotlib charts used in the **PDF** (`angle_vs_time`, `score_per_rep`,
  `rep_durations`, `check_breakdown`) and interactive Altair charts used on the **Results page**
  (`ui_score_per_rep`, `ui_rep_durations`, `ui_angle_vs_time`).

### `ui/styles.py`
* **Purpose:** the CSS design system (colours, fonts, layout of custom HTML) and small HTML
  constants (`NAV_ACTIVE`, `HERO_BG`, `BRAND_MARK`).

### `utils/helpers.py`, `utils/export.py`, `utils/logger.py`
* `helpers.py`: `ensure_pose_model()` (downloads the model once), `is_supported_video()`,
  `format_seconds()`.
* `export.py`: `report_csv()`, `report_pdf()`, `DISCLAIMER`.
* `logger.py`: `get_logger()`.

### `tests/`
Seven files, 86 tests in total (some tests are parametrised), described in section 45.

---

# 5. File-by-file deep explanation

This section goes through the source files in pipeline order. Code snippets are copied from the
repository; long files are summarised.

## 5.1 `config/thresholds.py`

### Purpose
All tunable numbers. Nothing else in the code base contains a threshold literal that changes
analysis behaviour (the UI has layout numbers only).

### Imports
* `from pathlib import Path`: builds project-relative paths (`PROJECT_ROOT`, `MODEL_DIR`,
  `SAMPLE_DIR`, `OUTPUT_DIR`), so the project has no machine-specific paths.

### Contents, grouped

| Group | Key values | Meaning |
|---|---|---|
| Paths | `POSE_MODEL_PATH = models/pose_landmarker_lite.task`, `POSE_MODEL_URL` | where the model lives and where it is downloaded from |
| `SUPPORTED_VIDEO_TYPES` | `("mp4", "avi", "mov")` | accepted uploads |
| `PROCESSING` | `max_width 640`, `frame_skip 1`, `webcam_index 0`, `webcam_max_seconds 120` | performance defaults |
| `PREPROCESSING` | `dark_mean_threshold 70`, `low_contrast_std_threshold 30`, `clahe_clip_limit 2.5`, `clahe_tile_grid (8, 8)`, `noise_sigma_threshold 4.0`, `noise_sigma_threshold_dark 1.5`, `gaussian_kernel (5, 5)`, `blur_laplacian_threshold 40.0`, `mask_threshold 0.5`, `mask_morph_kernel 5`, `background_blur_kernel (31, 31)`, `roi_padding 0.15` | DIP decisions |
| `POSE` | `num_poses 2`, detection/presence/tracking confidence `0.5`, `landmark_visibility_threshold 0.5`, `max_missing_seconds 1.5` | model settings |
| `FEATURES` | `ema_alpha 0.5`, `median_window 5`, `side_switch_margin 0.15`, `frontal_view_ratio 0.45` | smoothing and side/view |
| `RECOGNITION` | `window_seconds 2.0`, `min_samples 8`, `horizontal_body_deg 50`, `min_rom_deg 15`, `rom_full_deg 40`, `min_confidence 0.5`, `vote_frames 15` | recognition cues |
| `SCORE_BANDS` | 100 Excellent detected form, 90 Very good, 75 Acceptable, 60 Needs improvement, 0 Poor detected form | grade labels |
| `STATUS_GOOD`, `STATUS_WARNING` | 75, 50 | traffic-light colours and "is this an issue?" |
| `TEMPO` | ideal 1.0 s, zero 0.3 s | tempo check for all exercises |
| `EXERCISE_THRESHOLDS` | per exercise `rep` (extended, flexed, min_rep_seconds) and `checks` (ideal, zero) | see sections 18-21 |
| `COLORS` | BGR tuples | overlay colours (OpenCV uses BGR order) |

## 5.2 `core/angle_calculator.py`

### Imports
* `math`: `hypot` (vector length), `acos`, `degrees`.
* `typing.Sequence`: the `Point` type alias (anything indexable with `[0]`, `[1]`).

### Function: `calculate_angle(point_a, point_b, point_c)`
1. **What it does:** returns the angle ABC in degrees, with B as the joint (vertex).
2. **Why it exists:** every joint angle (knee, hip, elbow, shoulder, ankle, body line) is computed
   with it.
3. **Parameters:** three points `(x, y)`.
4. **Return value:** a float between 0 and 180.
5. **Mathematical logic:** vectors BA = A − B and BC = C − B; θ = arccos(BA·BC / (|BA||BC|)).
   The cosine is clamped to [−1, 1] before `acos` because floating-point rounding can produce
   1.0000000002, which would crash `acos`.
6. **Where it is called:** `core/feature_extractor.py → FeatureExtractor.extract()` (knee, hip,
   ankle, elbow, shoulder, body-line angles), `angle_from_vertical()` (below),
   `ui/figures.py → joint_angle()`.
7. **Example input:** hip A = (300, 200), knee B = (330, 290), ankle C = (300, 380).
8. **Example output:** BA = (−30, −90), BC = (−30, 90); dot = 900 − 8100 = −7200;
   |BA| = |BC| = √9000 ≈ 94.87; cos θ = −7200 / 9000 = −0.8; θ = arccos(−0.8) ≈ **143.1°**.

Raises `ValueError` if A or C equals B (a zero-length vector has no direction). The feature
extractor catches that and returns `None` for the feature.

### Function: `angle_from_vertical(start, end, upward=True)`
* **What it does:** angle between the segment start→end and the vertical direction.
* **How:** builds a reference point one pixel above (`upward=True`) or below (`upward=False`) the
  start point and reuses `calculate_angle(reference, start, end)`.
* **Remember:** image y grows **downward**, so "one pixel above" is `y − 1`.
* **Used for:** `torso_inclination` (hip→shoulder, 0° = upright), `body_inclination`
  (ankle→shoulder, 0° = standing, 90° = lying), `upper_arm_elevation` (shoulder→elbow with
  `upward=False`, 0° = arm hanging down), `forearm_from_vertical` (elbow→wrist, 0° = forearm
  pointing straight up).
* **Example:** hip (100, 300), shoulder (130, 200): the segment leans 30 px forward over 100 px up;
  angle ≈ arctan(30/100) ≈ **16.7°**.

### Function: `distance(p, q)`
Euclidean distance √((px−qx)² + (py−qy)²) via `math.hypot`. Used for body-segment lengths
(torso, thigh, shin) that normalise other features.

### Function: `signed_distance_to_line(p, a, b)`
* **What it does:** the perpendicular distance of point `p` (the hip) from the line a→b
  (shoulder→ankle), **divided by the line length**, with a sign.
* **Maths:** the 2-D cross product (b−a) × (p−a) = (bx−ax)(py−ay) − (by−ay)(px−ax) equals
  |ab| × (perpendicular distance). Dividing by |ab|² gives distance ÷ |ab|, a unitless fraction
  of body length.
* **Sign:** the cross-product sign depends on the direction of the line, so the function flips it
  when the line points left (`b[0] < a[0]`). Result: **positive = below the line** (larger y) =
  sagging hips; negative = above the line = piked hips.
* **Used for:** `hip_line_offset` (push-up body alignment).
* **Example:** shoulder a = (400, 300), ankle b = (100, 330), hip p = (250, 335). The line points
  left, so the sign is flipped. cross = (−300)(35) − (30)(−150) = −10500 + 4500 = −6000;
  |ab|² = 90000 + 900 = 90900; d = −0.066; flipped → **+0.066** (hips 6.6 % of body length below
  the line → sagging).

## 5.3 `core/pose_detector.py`

### Imports
* `numpy`: landmark arrays.
* `mediapipe` (imported **inside** `PoseDetector.__init__`, not at module level): the model is only
  loaded when a detector is actually created, so starting the app or opening the Home page does
  not load MediaPipe.
* `from mediapipe.tasks.python import BaseOptions, vision`: the Tasks API
  (`PoseLandmarkerOptions`, `PoseLandmarker`, `RunningMode`).
* `config.thresholds.POSE, POSE_MODEL_PATH`; `utils.helpers.ensure_pose_model`;
  `utils.logger.get_logger`.

### Constants
* `LANDMARK_NAMES`: the 33 MediaPipe landmark names in model order (0 `nose` … 32
  `right_foot_index`).
* `LM`: dictionary name → index, e.g. `LM["left_knee"] == 25`.
* `SKELETON`: 18 connections between body landmarks (face points are omitted): shoulders,
  both arms, torso sides, hips, both legs, and each foot as a triangle (ankle, heel, foot index).

### Class: `PoseDetection` (dataclass)
Attributes: `points` (33×2 float32 array of pixel coordinates or `None`), `visibility` (33
values in [0, 1]), `num_people`, `mask` (H×W probability map or `None`), `error` (text or
`None`), `all_boxes` (bounding boxes of every detected person). Property `detected` is
`points is not None`.

### Function: `prominence(points, visibility)`
Area of the bounding box of visible landmarks (visibility ≥ 0.5) multiplied by mean visibility.
The person with the largest value is analysed. Reason: the person exercising close to the camera
is usually the biggest and clearest.

### Class: `PoseDetector`
* **`__init__(with_segmentation=False, num_poses=None)`**
  1. `ensure_pose_model()` downloads `pose_landmarker_lite.task` if missing.
  2. Builds `PoseLandmarkerOptions`: model path, `running_mode=VIDEO`, `num_poses=2`,
     `min_pose_detection_confidence=0.5`, `min_pose_presence_confidence=0.5`,
     `min_tracking_confidence=0.5`, `output_segmentation_masks=with_segmentation`.
  3. `PoseLandmarker.create_from_options()` loads the model (this is the slow step, done once per
     session).
* **`detect(rgb, timestamp_ms)`**
  1. Forces timestamps to increase strictly (`max(timestamp, last + 1)`), because VIDEO mode
     rejects repeated or decreasing timestamps.
  2. Wraps the NumPy array in `mp.Image(image_format=SRGB)`.
  3. `detect_for_video(image, timestamp_ms)` runs inference.
  4. Any exception is caught and returned as `PoseDetection(error="Pose detector failure: …")`,
     so one bad frame never crashes the session.
  5. For each detected person, converts normalised coordinates to pixels:
     `(lm.x * w, lm.y * h)`, reads `lm.visibility`, computes `prominence`.
  6. Picks the most prominent person, and if segmentation is enabled, that person's mask.
* **`close()`** releases the native model resources. Every caller closes the detector in a
  `finally:` block.

## 5.4 `processing/preprocessing.py` (stage 1)

Full DIP details are in section 10. Summary of the functions:

| Function | Does | OpenCV / NumPy calls |
|---|---|---|
| `resize_frame(frame, max_width)` | downscale to width ≤ `max_width`, keep aspect ratio, never upscale | `cv2.resize(..., interpolation=cv2.INTER_AREA)` |
| `grey_statistics(grey)` | mean and standard deviation of grey levels | `cv2.meanStdDev` |
| `estimate_noise_sigma(grey)` | Immerkær noise estimate | `cv2.filter2D` with `_NOISE_KERNEL`, `np.abs().sum()` |
| `apply_clahe(bgr)` | CLAHE on the L channel of LAB | `cv2.cvtColor(BGR2LAB)`, `cv2.split`, `cv2.createCLAHE`, `cv2.merge`, `cv2.cvtColor(LAB2BGR)` |
| `preprocess_frame(frame_bgr, roi, max_width, enhance)` | the whole chain, decides what to apply | all of the above + `cv2.GaussianBlur`, `cv2.cvtColor(BGR2GRAY / BGR2RGB / GRAY2BGR)` |

`PreprocessResult` carries `bgr`, `rgb`, `scale`, `brightness`, `contrast`, `noise_sigma`,
`applied` (e.g. `["resize", "gaussian_denoise", "clahe"]`) and a `poor_lighting` property.

## 5.5 `processing/image_processing.py` (stage 3)

| Function | Does |
|---|---|
| `landmarks_roi(points, visibility, shape, min_visibility)` | bounding box of visible landmarks padded by 15 % of its width/height, clipped to the image; `None` if fewer than 4 visible landmarks |
| `laplacian_edges(grey)` | `cv2.Laplacian(grey, cv2.CV_64F, ksize=3)` |
| `sharpness(bgr, roi)` | variance of the Laplacian inside the ROI |
| `binary_person_mask(probability)` | threshold at 0.5 → {0, 255}, then morphological opening and closing with a 5×5 elliptical element |
| `suppress_background(bgr, mask)` | blur (31×31) and halve the background, feather the mask (7×7 blur) and alpha-blend |
| `edge_map_for_display(bgr)` | absolute Laplacian, min-max normalised to 0-255 for viewing (DIP viewer only) |

## 5.6 `core/feature_extractor.py` (stage 4)

### Imports
* `dataclasses`: `Features`.
* `numpy`: visibility means, arrays.
* `config.thresholds.FEATURES, POSE`.
* The four geometry functions from `core/angle_calculator.py`.
* `core.pose_detector.LM, PoseDetection`.

### Class: `Features`
`values` (feature name → float or `None`), `side` (`"left"`/`"right"`), `view`
(`"side"`/`"front"`), method `get(name)`.

### Class: `FeatureExtractor`
* **State:** `alpha` (EMA factor 0.5), `side` (starts `"left"`), `_side_vis` (smoothed
  visibility per side), `_ema` (last smoothed value per feature).
* **`_update_side(vis)`:** for each side, mean visibility of shoulder, elbow, wrist, hip, knee,
  ankle; smoothed with 0.7·old + 0.3·new. Switches side only if the other side is better by more
  than 0.15 (hysteresis, prevents flicker).
* **`_smooth(name, value)`:** EMA `y_t = α·x_t + (1−α)·y_(t−1)`. A `None` value clears that
  feature's memory, so smoothing restarts cleanly when the joint reappears.
* **`reset()`:** clears all EMA memory (called when no person is detected).
* **`extract(pose)`:** computes the 15 features (full list and formulas in section 16),
  detects front view, smooths, returns `Features`.

Inner helpers: `p(name, side)` returns a landmark point only if its visibility ≥ 0.5, otherwise
`None`; `safe(fn, *points)` returns `None` if any point is missing or the geometry is degenerate.
This is how **missing landmarks turn into `None` features instead of wrong numbers**.

## 5.7 `core/exercise_recognizer.py` (stage 5)

Detailed in section 17. Key items: `_ramp(x, lo, hi)` (clipped linear ramp), `Recognition`
dataclass, `ExerciseRecognizer.classify(samples)` (static, pure function of a window), and
`ExerciseRecognizer.update(t, features)` (sliding window + majority vote).

## 5.8 `core/rep_counter.py` (stage 6)

Detailed in section 22. `RepState` enum (`WAITING`, `START`, `TO_TURN`, `TURN`, `TO_START`),
`RepEvent` dataclass, `RepCounter` class with `_zone`, `_further`, `in_rep`, `reset`, `update`.

## 5.9 `exercises/base_exercise.py`

### Class: `FormCheck` (frozen dataclass)
| Attribute | Meaning |
|---|---|
| `key` | id of the (ideal, zero) pair in `EXERCISE_THRESHOLDS` |
| `label` | name shown in the UI |
| `feature` | which feature to measure (or `DURATION` = `"_rep_duration"`) |
| `reduce` | `"min"`, `"max"` or `"range"`: how the rep's per-frame values become one number |
| `message` | feedback template with `{value}` |
| `weight` | contribution to the rep score (default 1.0) |
| `live` | also evaluate on single frames for real-time colouring |

`TEMPO_CHECK = FormCheck("tempo", "Control (tempo)", DURATION, "max", "Rep was fast ({value:.1f} s): slow down …", weight=0.5)` is appended to every exercise.

### Class: `Exercise` (frozen dataclass)
`key`, `name`, `rep_feature`, `start_extended`, `phase_names` (4 labels), `checks`,
`display_features`, `description`. Properties/methods: `all_checks` (checks + tempo),
`rep_thresholds` (from config), `check_range(check)` (the (ideal, zero) pair).

## 5.10 `core/form_analyzer.py` (stage 7)

* `_REDUCERS = {"min": min, "max": max, "range": max − min}`.
* `median_filter(series, window=5)`: running median, edges use a shrunken window.
* `RepResult`: `index`, `exercise`, `kind`, `start_time`, `end_time`, `values`, `scores`,
  `score`, `issues`, property `duration`.
* `LiveCheck`: `check`, `value`, `score`.
* `FormAnalyzer.evaluate_rep(frames, start_time, end_time, index, kind)` and
  `FormAnalyzer.live(features)`; see section 23.

## 5.11 `core/scoring.py` (stage 8)

See section 24. Four small functions: `check_score`, `weighted_score`, `grade`, `status`.

## 5.12 `core/pipeline.py`

### Class: `ExerciseTracker`
One per exercise that is being tracked. Holds a `RepCounter` built from the exercise's
thresholds, a `FormAnalyzer`, `_start_frames` (a `deque(maxlen=60)` of feature dicts recorded
while resting in the start position) and `_rep_frames` (feature dicts during the movement).

* **`phase`** maps the counter state to the exercise's phase names:
  `WAITING → "GET IN POSITION"`, `START → phase_names[0]`, `TO_TURN → [1]`, `TURN → [2]`,
  `TO_START → [3]`.
* **`update(t, values)`** feeds the rep feature to the counter. While no event occurs it stores
  the frame in the start buffer or the rep buffer. When a `RepEvent` arrives it evaluates
  `start frames + rep frames + current frame` with `FormAnalyzer.evaluate_rep()` and returns the
  `RepResult`. The start frames are included on purpose: checks such as "full extension before
  the curl" need the frames *before* the movement begins.

### Class: `FrameStatus`
Everything the UI needs for one frame: `t`, `pose_detected`, `num_people`, `mode`, `exercise`,
`recognition`, `reps`, `phase`, `status`, `live_checks`, `last_rep`, `feedback`, `warnings`,
`fps`, `features`, `pre`, `pose`, `sharpness`.

### Class: `AnalysisPipeline`
* **`__init__(detector, source, manual_exercise, enhance, background_focus, max_width)`**
  validates `manual_exercise` (raises `ValueError` for an unknown key), creates a
  `FeatureExtractor`, an `ExerciseRecognizer`, and **trackers**: only the chosen exercise in
  manual mode, **all four exercises in auto mode**. Creates the `SessionLog` with
  `mode="manual"` or `"auto"`.
* **`process(frame_bgr, t)`** runs the 11 steps listed in section 3 and returns
  `(annotated, FrameStatus)`.
* **`_compose_feedback(st, exercise)`**, **`_collect_warnings(...)`**, **`_log(...)`**: see
  sections 23, 30 and 28/29.
* **`rep_count(exercise)`** counts that exercise's reps in the session log.

An important detail (confirmed by running the code): **in auto mode all four trackers run in
parallel from the first frame**, and a finished rep is only recorded if its exercise is the
currently *active* (recognised) one. That is why a squat started before the recognizer has made
up its mind is still counted correctly once recognition succeeds (see section 33).

## 5.13 `core/session.py`

`SessionLog(source, mode, started_at, frames, reps, partials)`:

| Method | Returns |
|---|---|
| `duration` | last frame time − first frame time |
| `frames_df()` | one row per analysed frame (all logged columns) |
| `reps_df()` | one row per counted rep: rep, exercise, start/end/duration, score, issues, `score_<check>` and `value_<check>` columns |
| `exercises_done()` | sorted exercise keys that have at least one counted rep |
| `check_breakdown(exercise)` | average score per check across that exercise's reps |
| `angle_statistics()` | min/max/mean/std/samples of every feature over frames with a value |
| `issue_counts()` | how many reps scored below 75 on each check |
| `summary()` | the headline dict: date, source, mode, exercises, total/partial reps, average score, grade, best/worst rep, duration, frames analysed, pose detection rate |

## 5.14 `processing/visualization.py`

See section 25.

## 5.15 `processing/video_processor.py`

See sections 9 and 26.

## 5.16 `utils/`

* `ensure_pose_model(path)`: returns the path if the file exists and is non-empty; otherwise
  downloads it with `urllib.request.urlretrieve`; on failure deletes the partial file and raises
  `RuntimeError` with manual instructions.
* `is_supported_video(filename)`: checks the extension against `SUPPORTED_VIDEO_TYPES`.
* `format_seconds(seconds)`: `m:ss`.
* `report_csv(session)`: one long table with columns section, item, metric, value (summary,
  per-rep values, form issues, angle statistics, disclaimer).
* `report_pdf(session)`: A4 text pages via Matplotlib (`PdfPages`) followed by the Matplotlib
  charts from `ui/charts.py`.
* `get_logger(name)`: configures logging once (INFO level) and returns a module logger.

## 5.17 `demo/synthetic_athlete.py`

See section 27.

## 5.18 `ui/` files

See sections 7, 8 and 34 for behaviour. Implementation notes:

* `ui/components.py → md(markup)` is `st.markdown(markup, unsafe_allow_html=True)`: custom HTML
  for panels and read-outs. `css(markup)` is `st.html(markup)`, used for style-only HTML so it
  takes no layout space.
* `ui/figures.py` samples one scripted good-form rep per exercise (`rep_cycle(key, n=24)`),
  converts the athlete's joints to MediaPipe landmark names, draws the `SKELETON` connections,
  computes angles with `calculate_angle`, and runs the real `ExerciseTracker` state machine over
  the sampled angles to label phases. The motion is SVG `<animate>` over the sampled frames.
* `ui/charts.py`: the four Matplotlib functions are used only by the PDF; the three `ui_*`
  Altair functions are used on the Results page (with hover tooltips).

---

# 6. Application startup

What happens when you run `streamlit run app.py`:

1. **Streamlit starts a local web server.** `.streamlit/config.toml` binds it to `localhost`
   (so the webcam app is not exposed to your network), sets the maximum upload size (500 MB), and
   sets the theme (dark background, coral-red accent, Archivo / IBM Plex fonts). The browser
   opens `http://localhost:8501`.
2. **When the browser connects, Streamlit executes `app.py` from top to bottom.** It re-executes
   the whole script on every interaction (button click, widget change). Python modules are
   imported once per server process and then cached.
3. **Imports on the first run:** `streamlit`, then `ui.pages`, which imports `cv2`, `pandas`,
   `config.thresholds`, `core.pose_detector` (but **not** MediaPipe itself, which is imported
   lazily inside `PoseDetector.__init__`), `core.pipeline` (and through it the feature
   extractor, recognizer, rep counter, form analyzer, scoring, session, exercises, preprocessing,
   image processing), `demo.synthetic_athlete`, `processing.video_processor`, `ui.charts`
   (which sets Matplotlib's non-interactive `Agg` backend and imports Altair),
   `ui.figures`, `ui.components`, `ui.styles`, `utils.export`, `utils.helpers`.
4. **`st.set_page_config(...)`** sets the browser tab title "FORM/AI - Gym form analysis", the
   icon and the wide layout.
5. **`css(CSS)`** injects the design-system stylesheet from `ui/styles.py`.
6. **Page selection:** `st.query_params.get("page", "Home")` reads `?page=...` from the URL; if
   valid, it becomes the initial value of `st.session_state["page"]` (only on the first run of a
   browser session, because `setdefault` is used).
7. **Settings:** `read_settings()` makes sure every settings key exists in session state with its
   default (`exercise_choice="auto"`, `max_width=640`, `frame_skip=1`, `enhance=True`,
   `focus=False`, `cam=0`, `cam_secs=120`, `mirror=True`) and returns a `Settings` object.
8. **`pages.top_nav()`** draws the navigation bar.
9. **`PAGE_FUNCS[page](settings)`** draws the chosen page. The call is wrapped in
   `try/except RuntimeError` so problems such as "pose model missing and download failed" show as
   an error message instead of a crash.
10. **`pages.footer()`** draws the disclaimer.

**Which models are loaded at startup?** None. The MediaPipe model is loaded only when a
`PoseDetector` is created: when you press **Start** on the camera page, **Analyse video**,
**Analyse this clip**, **Run batch validation**, or open the DIP pipeline viewer. Each of these
creates a fresh detector for that session and closes it afterwards.

**When is the camera accessed?** Only after you press **Start** on the Analyze → Camera page
(`ui/pages.py → live()` → `open_webcam()`), and it is released in a `finally:` block when the
session ends or you press **Stop**.

**What does the Home page compute?** The skeleton illustrations. `ui/figures.py` runs the
scripted athlete and the real rep state machine (pure Python maths, no model), cached with
`functools.lru_cache` so this happens once per server process.

---

# 7. Streamlit UI

## 7.1 How Streamlit works (the one idea you must know)

Streamlit re-runs `app.py` **from the top on every interaction**. There is no "event loop" in
your code. A button returns `True` only during the run triggered by clicking it, and anything that
must survive between runs is kept in `st.session_state` (a per-browser-tab dictionary). Buttons in
this project mostly use `on_click=` callbacks, which run *before* the script re-executes, so the
new page is drawn in the same rerun.

Internally the app still uses seven page names (the keys of `PAGE_FUNCS` in `app.py`):
`Home`, `Live Analysis`, `Video Upload`, `Demo Mode`, `Exercise Analysis`, `Results`, `About`.
The navigation bar groups `Live Analysis`, `Video Upload` and `Demo Mode` under **Analyze**, and
labels `Exercise Analysis` as **Exercises**.

## 7.2 Home (`ui/pages.py → home()`)

**What you see**
* The navigation bar (on every page).
* The hero: the label "Pose-based form analysis", the headline "Every rep, measured.", one
  sentence of explanation, the buttons **Start analysis** and **Try a demo clip**.
* On the right, an animated skeleton performing a squat with a live read-out of knee angle, hip
  angle and phase, a dashed hip/knee trajectory, the "SUBJECT ROI" corner brackets and a
  normalised x-axis. Faint background studies of a push-up, curl, squat and press.
* A band of four numbers computed from the code: `len(LANDMARK_NAMES)` = 33,
  `len(EXERCISES)` = 4, total form checks = 20 (5 per exercise including tempo), "1 camera".
* Four exercise cards (animated skeleton, rep signal, measured range, number of checks) and the
  link **How each lift is scored →**.

**Which code creates it**
* `css(HERO_BG.format(uri=figures.studies_background()))` sets the background studies as a CSS
  data URI on the hero container.
* `st.columns([5, 6])` splits the hero; the text is custom HTML via `md()`; the buttons are
  `st.button`s in a horizontal container.
* `figures.hero_svg()` generates the right-hand figure; `figures.exercise_svg(key)` the cards.

**Important:** the figures are **not** decorative drawings. `ui/figures.py → rep_cycle()`
samples the good-form scripted rep of each exercise from `demo/synthetic_athlete.py` (the same
model that renders the demo clips), `joint_angle()` measures angles with
`core/angle_calculator.calculate_angle`, and the phase labels come from running the real
`core/pipeline.ExerciseTracker` state machine over those angles. The bones are the real
`core/pose_detector.SKELETON` connections. The "SUBJECT ROI" box is computed by
`processing/image_processing.landmarks_roi`.

**Buttons:** see section 34.

## 7.3 Analyze: Camera (page key `Live Analysis`, `ui/pages.py → live()`)

**What you see**
* Header "ANALYZE" and a segmented source switcher **Camera | Video file | Demo clips**
  (`_analyze_header()`).
* The **Exercise** segmented control: **Auto detect | Squat | Bicep Curl | Push-Up |
  Shoulder Press**, and a **Processing** popover with *Processing width*, *Adaptive
  enhancement*, *Background focus*, *Camera index*, *Max session length*, *Mirror view*
  (`ui/components.py → settings_bar(camera=True)`; the frame-skip slider is hidden here because it
  only applies to files).
* Left: the camera "stage". Before starting it shows a framing guide
  (`figures.camera_placeholder_svg()`, the squat start pose inside its ROI) with
  "CAMERA 0 · STANDBY" and the mode.
* Right: **Start** / **Stop** buttons and the read-out panel. Before starting it is the
  `idle_panel()` with setup instructions.
* If a previous camera session exists: "Last camera session" with the headline numbers
  (`session_overview()`) and **Open full report**.

**While running** the left side shows the annotated frame (`frame_slot.image(annotated,
channels="BGR")`) and the right side `status_panel(stt, display_fps)`: exercise name, detection
mode and confidence, status tag (GOOD FORM / CHECK FORM / CORRECT YOUR FORM / WAITING), REPS,
PHASE, LAST REP score, "Posture, this frame" live checks (PASS / WATCH / FIX with score),
"Last rep" breakdown for every check, Feedback messages, Warnings, and telemetry (processing FPS,
display FPS, people detected).

## 7.4 Analyze: Video file (page key `Video Upload`, `ui/pages.py → upload()`)

**What you see**
* The same header and settings (the Processing popover now contains *Analyse every n-th video
  frame* instead of the camera options).
* A file uploader "Exercise video" accepting mp4, avi, mov.
* After choosing a file: a line with file name, frame count, FPS and duration, and the
  **Analyse video** button.
* During analysis: annotated preview (left), read-out panel (right), a progress bar.
* After analysis: "Result" with headline numbers, the annotated output video (expander) and
  **Open full report**.
* Always (once a file is loaded): the expander **"DIP pipeline viewer: inspect every processing
  stage on one frame"** with a time slider.

## 7.5 Analyze: Demo clips (page key `Demo Mode`, `ui/pages.py → demo()`)

**What you see**
* Header and settings.
* Left: the selected demo clip as a video player.
* Right: the **Demo clip** dropdown (10 scenarios, e.g. `squat_bad - Shallow squats with
  excessive forward lean`), a "Scripted ground truth" panel (exercise, number of reps, scenario),
  a note explaining the clips are synthetic but analysed by the real pipeline, and
  **Analyse this clip**.
* After analysis: the same result block as for uploads.
* Section "Validate on all demo clips" with **Run batch validation**, which shows a table of
  expected vs detected exercise and reps for all 10 clips.

## 7.6 Exercises (page key `Exercise Analysis`, `ui/pages.py → exercise_guide()`)

**What you see**
* A segmented control to choose the lift (`key="guide_ex"`, default squat).
* Left: the looping skeleton of the scripted good rep (`figures.exercise_svg`) and the start
  and turn positions side by side with the measured rep angle at each
  (`figures.pose_pair_svg`).
* Right: name, description, rep signal, primary joint, start zone and turn zone (from
  `EXERCISE_THRESHOLDS[...]["rep"]`), the four phase names, and "signals measured".
* A table of all form checks with the feature, reduction (min/max/range), ideal (100) and
  zero (0) values, weight and whether it is a live check (`_checks_table()`).
* The score bands as a coloured scale and the tempo rule.

No analysis runs on this page; it only reads definitions from `exercises/` and
`config/thresholds.py`.

## 7.7 Results (`ui/pages.py → results()`)

**With no session:** an empty state (skeleton + "No analysis yet…") and three buttons
**Demo clips**, **Video file**, **Camera**.

**With a session** (`st.session_state["last_session"]`):
* Title "SESSION REPORT" and a metadata line: source, mode, date/time, frames, person-detection
  rate.
* Headline strip (`kpi_strip()`): **Total reps** (+ partial reps not counted), **Average form
  score** (+ grade, coloured), **Best rep** (+ its score), **Session time** (+ worst rep).
* A warning if the person was detected in less than 60 % of frames.
* **Form breakdown** (`_breakdown()`): average score per check for each exercise, drawn as thin
  coloured bars (from `SessionLog.check_breakdown()`).
* **Score per rep** (`charts.ui_score_per_rep`): bars coloured by band, dashed lines at 75 and 50,
  hover tooltip with grade, duration and issues.
* For each exercise, **<rep feature> over time** (`charts.ui_angle_vs_time`): the rep angle vs
  time, counted reps shaded, dashed threshold lines "extended" and "flexed", hover read-out.
* Tabs: **Repetitions** (the `reps_df()` table), **Rep timing** (bar chart of movement time),
  **Form errors** (issue counts + per-rep messages), **Joint-angle statistics**
  (`angle_statistics()`), **Export** (three CSV downloads, **Build PDF report**, then
  **Download report (PDF)**).

## 7.8 About (`ui/pages.py → about()`)

Static content: problem statement, objectives, the pipeline as a chain of stages, the eight DIP
concepts used (`DIP_CONCEPTS`), technologies, limitations, future improvements. No processing.

---

# 8. Navigation

## 8.1 How it is implemented

Navigation is **custom**, not Streamlit's built-in multipage system (`st.navigation` / `pages/`
folder is **not used**) and not a sidebar (the sidebar is not used at all; CSS in `ui/styles.py`
also hides it).

* The current page is **one string in session state**: `st.session_state["page"]`.
* `ui/pages.py → top_nav()` draws a horizontal container with the brand, five `st.button`s
  (`type="tertiary"`, keys `nav_home`, `nav_analyze`, `nav_exercises`, `nav_results`,
  `nav_about`) and the **Start analysis** button (`nav_cta`).
* Every navigation button has `on_click=goto, args=(target_page,)`.
* `goto(page)` does two things:
  ```python
  def goto(page: str) -> None:
      st.session_state["page"] = page
      st.query_params["page"] = page
  ```
  It stores the page and mirrors it into the URL (`?page=...`), so a browser refresh or a shared
  link opens the same page.
* `app.py` then calls `PAGE_FUNCS[st.session_state["page"]](settings)`.
* The active link is highlighted by injecting a tiny CSS rule (`NAV_ACTIVE`) that targets the
  button whose key matches the current page (Streamlit gives keyed widgets a CSS class
  `st-key-<key>`).

## 8.2 The Analyze source switcher

`_analyze_header(page)` sets `st.session_state["source_choice"]` to the label of the current page
("Camera", "Video file" or "Demo clips") **before** drawing the segmented control, so the control
always shows the right selection. Its `on_change=_source_changed` callback maps the chosen label
back to the page key and calls `goto()`.

## 8.3 Why settings survive page changes

Streamlit deletes a widget's session-state value when that widget is not drawn in a run. The
settings widgets only exist on the Analyze pages, so on every run
`ui/components.py → read_settings()` does:

```python
for key, default in SETTING_DEFAULTS.items():
    ss[key] = ss.get(key, default)   # re-assign so Streamlit keeps it while the widget is hidden
```

Re-assigning turns the value into ordinary session state, which is never cleaned up. This is
tested by `tests/test_app.py → test_manual_mode_is_applied_and_kept_across_pages`.

---

# 9. Image input

## 9.1 Webcam

```text
Webcam hardware
  ↓  cv2.VideoCapture(index, backend)        processing/video_processor.py → open_webcam()
     (Windows: DirectShow first, then any backend; the first frame is read to prove it works)
  ↓  cap.read() in a loop                     ui/pages.py → live()
  ↓  cv2.flip(frame, 1) if "Mirror view"      (selfie view)
  ↓  t = time.perf_counter() - t0             (wall-clock seconds since Start)
  ↓  pipeline.process(frame, t)               core/pipeline.py
  ↓  preprocessing → pose detection → ...
```

Details from `live()`:
* The loop runs while `time.perf_counter() - t0 < settings.webcam_seconds` (default 120 s,
  adjustable 15-600 s).
* If `cap.read()` fails more than 30 times in a row, it shows "The camera stopped delivering
  frames" and stops.
* The display FPS is smoothed: `0.9 * old + 0.1 * instantaneous`.
* The capture and the detector are released in `finally:`, so pressing **Stop** (which makes
  Streamlit abandon the running script) still frees the camera.
* The timestamp is **real time**, because live frames arrive at whatever rate the camera and CPU
  allow.

## 9.2 Uploaded video

```text
Uploaded file (bytes in memory)
  ↓  saved to outputs/uploads/<name>                ui/pages.py → upload()
  ↓  video_info(path): can it be opened? read the first frame; FPS and frame count
  ↓  iterate_video(path, frame_skip)               processing/video_processor.py
       cv2.VideoCapture(path); cap.read() until the end;
       yield every frame_skip-th frame with t = frame_index / fps
  ↓  analyze_video(): pipeline.process(frame, t) for each yielded frame
  ↓  annotated frames → VideoWriter (outputs/<stem>_analyzed.webm)
  ↓  on_frame callback → live preview in the UI (every 3rd update)
```

* **Timestamps come from the frame index and the file's FPS**, not from how fast your computer
  runs. So a 16-second video always produces 16 seconds of analysis data, even if processing takes
  longer. If a file reports an invalid FPS, 30 is assumed.
* Demo clips go through exactly the same path (`run_video_analysis()` → `analyze_video()`).

## 9.3 Colour order

OpenCV delivers frames in **BGR** order (blue, green, red). The whole pipeline keeps BGR for
drawing and display; only the copy given to MediaPipe is converted to RGB
(`preprocess_frame()` → `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`). Streamlit is told the channel
order explicitly: `st.image(annotated, channels="BGR")`.

---

# 10. Image preprocessing

This is stage 1, `processing/preprocessing.py → preprocess_frame()`. The principle is
**measure first, apply only when needed**: good footage passes through unchanged (apart from
resizing), so the model is never fed an over-processed image.

The order inside `preprocess_frame()`:

```text
raw BGR frame
  │ (if greyscale source: GRAY → BGR)
  ▼
1. resize_frame()        downscale to width ≤ max_width (INTER_AREA)
2. BGR → grey
3. grey_statistics()     brightness (mean), contrast (std) on the subject ROI
4. estimate_noise_sigma() noise σ on the whole grey frame
5. if enhance:
     needs_clahe = brightness < 70 or contrast < 30
     noise_limit = 1.5 if needs_clahe else 4.0
     if σ > noise_limit:  GaussianBlur 5×5        ("gaussian_denoise")
     if needs_clahe:      CLAHE on L of LAB       ("clahe")
6. BGR → RGB             model input
  ▼
PreprocessResult(bgr, rgb, scale, brightness, contrast, noise_sigma, applied)
```

## 10.1 Resizing (geometric transformation)

* **What it is:** changing the image size (number of pixels).
* **Maths:** each output pixel is computed from the input pixels it covers. With `INTER_AREA`
  (area interpolation) an output pixel is the **average of the input pixels that fall inside it**,
  weighted by overlap area. Averaging is a low-pass filter, which prevents aliasing (jagged edges
  and moiré) when shrinking.
* **Why:** a 1920×1080 frame has 6.75 times more pixels than 640×360. The pose model works on a
  small internal input anyway, so full resolution only costs time. Smaller frames = higher FPS.
* **Input:** raw frame of any size. **Output:** frame with width ≤ `max_width` (default 640,
  selectable 320/480/640/960 in the Processing menu), same aspect ratio. Frames already narrow
  enough are **not** upscaled (`scale = 1.0`).
* **Code:** `processing/preprocessing.py → resize_frame()`:
  `cv2.resize(frame, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)`.
* **Effect:** fewer pixels, slightly smoother; landmark coordinates are then in the resized
  frame's pixel space.

## 10.2 Colour-space conversion

Three conversions are used:

| Conversion | Where | Why |
|---|---|---|
| BGR → Grey | `preprocess_frame()`, `sharpness()` | statistics, noise and edges only need intensity. OpenCV computes Y = 0.299 R + 0.587 G + 0.114 B |
| BGR → LAB → BGR | `apply_clahe()` | LAB separates **L** (lightness) from **a**, **b** (colour). Equalising only L improves contrast without shifting skin and clothing colours |
| BGR → RGB | end of `preprocess_frame()` | MediaPipe expects RGB; OpenCV stores BGR |
| GRAY → BGR | start of `preprocess_frame()` | lets a greyscale video go through the colour pipeline |

## 10.3 Grey-level statistics (lighting analysis)

* **What it is:** the mean and standard deviation of pixel intensities.
* **Maths:** for N pixels with grey values I: mean μ = (1/N) Σ I; standard deviation
  σ = √((1/N) Σ (I − μ)²). μ measures **brightness** (0 = black, 255 = white), σ measures
  **global contrast** (how spread the histogram is).
* **Why:** to decide objectively whether the frame is too dark (μ < 70) or too flat (σ < 30).
* **Subject ROI:** the statistics are computed on the **person's bounding box from the previous
  frame** (`roi` argument) when it is larger than 8×8 pixels. So a bright window behind a person
  in shadow does not hide the fact that the person is badly lit. On the first frame (no ROI yet)
  the whole frame is used.
* **Code:** `grey_statistics()` uses `cv2.meanStdDev`.
* **Output:** `brightness`, `contrast` (logged per frame in `SessionLog.frames`).
* **Example (real run, `squat_bad`):** brightness ≈ 155-165, so no CLAHE is applied.

## 10.4 Noise estimation (Immerkær, 1996)

* **What it is:** an estimate of the standard deviation σ of Gaussian sensor noise from a single
  image.
* **Maths:** convolve the image with the mask

  ```text
  N = [ 1 -2  1
       -2  4 -2
        1 -2  1 ]
  ```

  N is the difference of two Laplacian approximations, so smooth image structure (edges,
  gradients) mostly cancels and what remains is dominated by noise. Then

  ```text
  σ ≈ √(π/2) · 1 / (6 (W − 2)(H − 2)) · Σ |I * N|
  ```

  (sum of absolute responses over the interior pixels). The constant √(π/2) converts a mean
  absolute value into a standard deviation for Gaussian noise; 6 normalises for the mask's gain.
* **Why:** to decide whether denoising is necessary, instead of always blurring.
* **Code:** `estimate_noise_sigma()` uses `cv2.filter2D(grey.astype(float32), -1, _NOISE_KERNEL)`,
  drops the one-pixel border, and applies the formula with NumPy.
* **Output:** `noise_sigma`. Real run on `squat_bad`: about 0.08-0.17 (a clean rendered video).

## 10.5 Gaussian filtering (denoising)

* **What it is:** a spatial low-pass filter that replaces each pixel by a weighted average of its
  neighbours, with weights from a 2-D Gaussian.
* **Maths:** G(x, y) = (1 / 2πσ²) · e^(−(x² + y²) / 2σ²). The 5×5 kernel is sampled from G and
  normalised to sum to 1; the output is the convolution I * G. With `sigmaX=0` OpenCV derives σ
  from the kernel size (σ = 0.3·((5−1)·0.5 − 1) + 0.8 = 1.1).
* **Why:** random sensor noise is high-frequency; the Gaussian removes it while keeping larger
  shapes (the body). It is applied **before** CLAHE because CLAHE amplifies noise.
* **When:** only if σ > 4.0, or σ > 1.5 when CLAHE is also about to run (dark frames
  under-report noise because values clip at 0, and CLAHE would amplify what is there).
* **Code:** `cv2.GaussianBlur(bgr, (5, 5), 0)` in `preprocess_frame()`; recorded as
  `"gaussian_denoise"` in `applied`.
* **Effect:** slightly softer image with far less grain; landmarks become stable in noisy
  low-light footage.

## 10.6 CLAHE (contrast enhancement)

* **What it is:** Contrast-Limited Adaptive Histogram Equalisation.
  * *Histogram equalisation* remaps intensities so the cumulative histogram becomes linear,
    spreading grey levels over the full range: new value = (L − 1) · CDF(old value).
  * *Adaptive*: done separately on 8×8 tiles, so each region gets its own mapping (a dark corner
    and a bright wall are both improved). Tile borders are bilinearly interpolated to avoid seams.
  * *Contrast-limited*: each tile's histogram is clipped at `clipLimit = 2.5` (times the average
    bin height) and the excess is redistributed. This limits the slope of the mapping, so noise in
    flat regions is not amplified hugely.
* **Why:** in dark or flat footage, body contours disappear and MediaPipe stops finding the
  person. CLAHE restores local contrast.
* **When:** only if brightness < 70 or contrast < 30 (`needs_clahe`) and "Adaptive enhancement"
  is on.
* **Code:** `apply_clahe()`: `cv2.cvtColor(BGR2LAB)` → `cv2.split` → `cv2.createCLAHE(2.5, (8, 8)).apply(L)` → `cv2.merge` → `cv2.cvtColor(LAB2BGR)`.
* **Effect:** shadows lifted, body edges visible, colours preserved.

**Measured effect** (`python -m demo.dip_experiment`, documented in README): on frames of
`squat_good` darkened to 10 % brightness with noise σ = 4, pose detection rate was 0.00 raw,
0.00 with CLAHE only, and **0.90** with the adaptive pipeline (denoise then CLAHE). At 6 %
brightness even the adaptive pipeline fails (0.00).

## 10.7 What preprocessing does **not** do

* **No pixel-value normalisation** (e.g. dividing by 255) in this project's code. MediaPipe
  performs its own input normalisation internally.
* **No histogram equalisation without CLAHE**, no gamma correction, no sharpening filter, no
  median filter on images (the median filter in the project is applied to *feature time series*,
  section 32).

---

# 11. Image processing pipeline (complete list)

| Technique | Implemented? | Where | Notes |
|---|---|---|---|
| Resize | **Yes** | `preprocessing.resize_frame()` | INTER_AREA, width ≤ 640 |
| Colour conversion | **Yes** | `preprocess_frame()`, `apply_clahe()`, `sharpness()` | BGR↔RGB, BGR→grey, BGR↔LAB, GRAY→BGR |
| Normalisation of pixel values | **No** (done inside MediaPipe) | | Coordinates and distances *are* normalised: section 15 |
| Min-max normalisation (display) | **Yes** | `image_processing.edge_map_for_display()` | only for the edge image in the DIP viewer |
| Gaussian filtering | **Yes** | `preprocess_frame()` (5×5, conditional); `suppress_background()` (31×31 background, 7×7 feathering) | |
| CLAHE | **Yes** | `apply_clahe()` | conditional |
| Noise estimation | **Yes** | `estimate_noise_sigma()` | Immerkær |
| Edge detection | **Yes** | `laplacian_edges()`, `sharpness()`, `edge_map_for_display()` | Laplacian only; **Canny/Sobel are not used** |
| Thresholding | **Yes** | `binary_person_mask()` | probability > 0.5 |
| Binary masks + morphology | **Yes** | `binary_person_mask()` | opening then closing, 5×5 ellipse |
| ROI extraction | **Yes** | `landmarks_roi()` | used for lighting statistics and sharpness |
| Background processing | **Yes, optional** | `suppress_background()` | only when "Background focus" is on (segmentation must be enabled) |
| Noise reduction | **Yes** | Gaussian (images), EMA and median (feature signals) | |

## 11.1 ROI extraction (`landmarks_roi()`)

```text
visible landmarks (visibility ≥ 0.5, at least 4 of them)
 → x1 = min x, y1 = min y, x2 = max x, y2 = max y
 → pad by 15 % of the box width / height on each side
 → clip to the image
 → (x1, y1, x2, y2), or None
```

Used for: lighting statistics on the **next** frame (`pipeline._roi` is passed to
`preprocess_frame`), sharpness on the current frame, the ROI rectangle in the DIP viewer, and the
"SUBJECT ROI" box in the home illustration.

## 11.2 Laplacian sharpness (`sharpness()`)

* **Laplacian:** ∇²I = ∂²I/∂x² + ∂²I/∂y², a second-derivative operator that responds strongly at
  edges and is near zero in flat regions. OpenCV approximates it with a 3×3 kernel (`ksize=3`).
* **Variance of the Laplacian:** a sharp image has many strong edge responses (high variance); a
  blurred image has weak ones (low variance).
* **Use:** if the value inside the ROI is below 40, the pipeline warns "Frame is blurry -
  landmarks may be less reliable". It does **not** change the image.
* Real run on `squat_bad`: about 3,500-6,000 (sharp rendered video).

## 11.3 Segmentation mask and morphology (`binary_person_mask()`)

Only when a detector is created with `with_segmentation=True` (the "Background focus" setting, or
the DIP viewer).

```text
MediaPipe probability map P(x, y) ∈ [0, 1]
 → threshold: M = 255 if P > 0.5 else 0          (segmentation by thresholding)
 → opening  = erosion then dilation              (removes isolated specks)
 → closing  = dilation then erosion              (fills small holes)
   structuring element: 5×5 ellipse (cv2.getStructuringElement(MORPH_ELLIPSE))
```

## 11.4 Background suppression (`suppress_background()`)

```text
background = GaussianBlur(frame, 31×31) // 2              (blurred and 50 % darker)
alpha      = GaussianBlur(mask, 7×7) / 255                (soft edge, 0..1)
output     = frame · alpha + background · (1 − alpha)     (alpha blending)
```

It affects only the **displayed/recorded** frame; pose detection has already run on the
unsuppressed frame.

## 11.5 DIP pipeline viewer (`ui/components.py → dip_stage_viewer()`)

On the Video file page you can pick any frame time and see: 1 resized input, 2 greyscale,
3 CLAHE, 4 Gaussian blur 5×5, 5 |Laplacian| edge map, 6 binary person mask, 7 landmarks +
skeleton + ROI rectangle, a histogram (input vs after CLAHE, `cv2.calcHist` with 64 bins), and
a table of the measurements with the decision the pipeline would make (CLAHE applied / denoise
applied / blur warning / multiple people). This viewer always shows CLAHE and blur so you can
*see* them; the real pipeline applies them only when the thresholds say so.

---

# 12. Pose estimation

## 12.1 What MediaPipe Pose is

MediaPipe is Google's library of pre-trained on-device vision models. This project uses the
**Pose Landmarker** task with the **BlazePose "lite"** model (`pose_landmarker_lite.task`,
float16, 5.8 MB). BlazePose is a two-stage deep-learning model:

1. a **detector** finds the person (region of interest) in the image;
2. a **landmark model** regresses 33 body keypoints inside that region.

In **VIDEO running mode** the landmarker tracks the person between frames: after the first
detection it reuses the previous frame's region, so the expensive detector does not run every
frame. That is why timestamps must increase.

## 12.2 Why it is used

* It runs in real time on a laptop **CPU**.
* It gives 33 landmarks including feet (heel, foot index), which the squat knee-tracking check
  needs.
* It gives a per-landmark **visibility** score, which the project uses to ignore unreliable
  joints.
* It can output a segmentation mask (used by Background focus and the DIP viewer).
* It installs with `pip`, no training needed.

## 12.3 How the model receives an image

```python
image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(rgb))
result = self._landmarker.detect_for_video(image, timestamp_ms)
```

* The input is the **preprocessed RGB frame** (≤ 640 px wide) from stage 1.
* `np.ascontiguousarray` makes sure the pixel memory is one contiguous block, as the native code
  requires.
* `timestamp_ms` = `int(t * 1000)` from `AnalysisPipeline.process()`, forced to be strictly
  increasing in `detect()`.

## 12.4 What the model produces

`result.pose_landmarks` is a list (one entry per detected person, up to `num_poses = 2`). Each
entry is a list of 33 landmarks. Each landmark has:

| Field | Meaning | Used in this project? |
|---|---|---|
| `x` | horizontal position, **normalised** 0..1 (0 = left edge, 1 = right edge) | **Yes** |
| `y` | vertical position, **normalised** 0..1 (0 = top, 1 = bottom) | **Yes** |
| `z` | relative depth (smaller = closer to camera), hip-centred | **No** (2-D analysis only) |
| `visibility` | likelihood the landmark is visible (not occluded), 0..1 | **Yes** (threshold 0.5) |
| `presence` | likelihood the landmark is present in the frame | No |

`result.pose_world_landmarks` (3-D coordinates in metres) is **not used**.
This is a planned improvement, not an implemented feature.

## 12.5 From normalised to pixel coordinates

```python
pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
vis = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
```

`w` and `h` are the width and height of the **processed** frame. Example: in a 640×480 frame a
knee at `x = 0.52, y = 0.71` becomes pixel (332.8, 340.8). All later geometry is in these pixel
coordinates, and the overlay is drawn on the same processed frame, so they line up.

## 12.6 Multiple people

With `num_poses = 2` the model can return two people. `prominence()` = bounding-box area of the
visible landmarks × mean visibility; the maximum is analysed. `num_people` is reported, the
pipeline warns "2 people detected - analysing the most prominent person only", and landmarks of
different people are never mixed. Demo clip `squat_two_people` tests this.

## 12.7 Visibility and confidence

* `min_pose_detection_confidence`, `min_pose_presence_confidence`, `min_tracking_confidence`
  (all 0.5) are MediaPipe's own acceptance thresholds.
* The project adds `landmark_visibility_threshold = 0.5`: any landmark below it is treated as
  **missing**; every feature that needs it becomes `None`
  (`FeatureExtractor.extract() → p()`), and the skeleton does not draw it.

## 12.8 Pose connections

`core/pose_detector.py → SKELETON` (18 connections):

```text
left_shoulder–right_shoulder
left_shoulder–left_elbow,   left_elbow–left_wrist
right_shoulder–right_elbow, right_elbow–right_wrist
left_shoulder–left_hip,     right_shoulder–right_hip,  left_hip–right_hip
left_hip–left_knee,         left_knee–left_ankle
right_hip–right_knee,       right_knee–right_ankle
left_ankle–left_heel,  left_heel–left_foot_index,  left_ankle–left_foot_index
right_ankle–right_heel, right_heel–right_foot_index, right_ankle–right_foot_index
```

Face landmarks (0-10) and hand landmarks (17-22: pinky, index, thumb) are detected by the model
but not drawn and not used in any feature.

## 12.9 The landmarks this project uses

| Landmark | Index (left / right) | Used for |
|---|---|---|
| Shoulder | 11 / 12 | elbow, hip, shoulder angles; torso, body inclination; wrist height; front-view detection |
| Elbow | 13 / 14 | elbow and shoulder angles; upper-arm elevation; forearm angle |
| Wrist | 15 / 16 | elbow angle; forearm angle; wrist above shoulder |
| Hip | 23 / 24 | knee, hip, shoulder angles; torso inclination; hip offset; hip above knee |
| Knee | 25 / 26 | knee and hip angles; knee over toe; knee width ratio |
| Ankle | 27 / 28 | knee and ankle angles; body inclination; hip offset; knee width ratio |
| Heel | 29 / 30 | which way the person faces (for knee over toe) |
| Foot index (toe) | 31 / 32 | ankle angle; knee over toe; facing direction |

Only **one side** of the body (the side facing the camera) is analysed at a time, chosen in
`FeatureExtractor._update_side()`. Both sides are used only for front-view detection and the knee
width ratio.

---

# 13. Pose landmark diagram

Side view (what the analysis mostly uses). The numbers are MediaPipe indices for the
**left** side; the right side is the same shape with indices 12, 14, 16, 24, 26, 28, 30, 32.

```text
                     ( 0 nose )           face points 0-10: detected, not used
                         │
      11 shoulder ●──────┘     (neck is not a landmark; no line is drawn)
                  │╲
                  │ ╲
                  │  ● 13 elbow
                  │   ╲
                  │    ● 15 wrist
  torso           │
  (hip→shoulder)  │
                  │
          23 hip  ●
                   ╲
                    ╲  thigh
                     ● 25 knee
                    ╱
                   ╱  shin
                  ● 27 ankle
                 ╱ ╲
       29 heel  ●───● 31 foot index (toe)
```

Front view (used for knee alignment):

```text
         12 ●────────────● 11          shoulders (distance vs torso length → view)
            │            │
            │            │
         24 ●────────────● 23          hips
            │            │
         26 ●            ● 25          knees      knee_width = |x25 − x26|
            │            │
         28 ●            ● 27          ankles     ankle_width = |x27 − x28|
```

Angles measured at each joint (vertex in the middle):

```text
knee_angle      = angle( hip,      KNEE,     ankle )
hip_angle       = angle( shoulder, HIP,      knee  )
ankle_angle     = angle( knee,     ANKLE,    toe   )
elbow_angle     = angle( shoulder, ELBOW,    wrist )
shoulder_angle  = angle( hip,      SHOULDER, elbow )
body_line_angle = angle( shoulder, HIP,      ankle )
```

---

# 14. Mathematics

## 14.1 Joint angle

**Given** three points: A (first), B (the joint), C (third). Example: A = hip, B = knee, C = ankle.

**Step 1, vectors from the joint:**

```text
BA = A − B = (Ax − Bx, Ay − By)
BC = C − B = (Cx − Bx, Cy − By)
```

**Step 2, dot product:** BA · BC = BAx·BCx + BAy·BCy. Geometrically
BA · BC = |BA| |BC| cos θ, where θ is the angle between the vectors.

**Step 3, magnitudes (lengths):** |BA| = √(BAx² + BAy²), |BC| = √(BCx² + BCy²)
(`math.hypot`).

**Step 4, solve for θ:**

```text
θ = arccos( (BA · BC) / (|BA| |BC|) )
```

**Step 5, radians to degrees:** `math.acos` returns radians (0..π); `math.degrees` multiplies by
180/π to get 0..180°.

**Why the result describes the joint:** the two vectors point along the two limb segments
(thigh and shin). The angle between them is the **included joint angle**: 180° is a straight
limb (standing, arm extended), smaller values mean more bending.

**Worked example (a knee in a partial squat):**

```text
A hip   = (300, 200)
B knee  = (330, 290)
C ankle = (300, 380)

BA = (300−330, 200−290) = (−30, −90)
BC = (300−330, 380−290) = (−30,  90)

BA · BC = (−30)(−30) + (−90)(90) = 900 − 8100 = −7200
|BA| = √(900 + 8100) = √9000 = 94.87
|BC| = √9000 = 94.87
cos θ = −7200 / (94.87 × 94.87) = −7200 / 9000 = −0.8
θ = arccos(−0.8) = 2.498 rad = 143.13°
```

A knee angle of 143° is between the squat thresholds (extended 160°, flexed 125°), so the state
machine would be in DESCENDING or ASCENDING.

**Why angles and not raw coordinates:** an angle does not change when the person moves across the
frame, stands closer or further away, or when the resolution changes. The same threshold (e.g.
"knee ≤ 125°") works for everyone.

**Clamping:** because of floating-point rounding, the cosine can be very slightly outside [−1, 1],
so the code clamps it: `max(-1.0, min(1.0, cosine))`.

## 14.2 Angle from the vertical

Reference point R one pixel straight up from the start S (or down, for hanging segments), then
angle(R, S, E). Equivalent to arccos of the vertical component of the unit vector S→E. Because
image y points **down**, "up" is `y − 1`.

## 14.3 Euclidean distance

d(P, Q) = √((Px − Qx)² + (Py − Qy)²). Used for torso length |hip–shoulder|, thigh length
|hip–knee| and shin length |knee–ankle|.

## 14.4 Signed perpendicular distance to a line

For line A→B and point P: cross = (Bx − Ax)(Py − Ay) − (By − Ay)(Px − Ax) = |AB| × signed
perpendicular distance. `signed_distance_to_line` returns cross / |AB|², i.e. perpendicular
distance **as a fraction of the line length**, with the sign fixed so that "below the line" is
positive. (Worked example in section 5.2.)

## 14.5 Clipped linear ramp (recognition)

```text
ramp(x, lo, hi) = clip( (x − lo) / (hi − lo), 0, 1 )
```

0 at `lo`, 1 at `hi`, linear between. Works in either direction (`lo > hi` gives a decreasing
ramp). `core/exercise_recognizer.py → _ramp()`.

## 14.6 Percentile range of motion

ROM = P95 − P5 of a joint angle over the 2-second window (`np.percentile`). Using the 5th and
95th percentiles instead of min and max ignores a few outlier frames.

## 14.7 Check score (linear interpolation, clipped)

```text
s = clip( (v − zero) / (ideal − zero), 0, 1 ) × 100
```

The sign of (ideal − zero) encodes direction: for squat depth ideal = 95, zero = 140, so smaller
knee angles are better; for push-up lockout ideal = 160, zero = 135, so larger is better.
`core/scoring.py → check_score()`.

## 14.8 Weighted mean (rep score)

```text
S = Σ (wᵢ · sᵢ) / Σ wᵢ      over checks with a measured score sᵢ
```

`core/scoring.py → weighted_score()`.

## 14.9 Exponential moving average

```text
S_t = α · X_t + (1 − α) · S_(t−1)      α = 0.5
```

Section 32.

## 14.10 Running median

For each position i: median of the values in [i − 2, i + 2] (window 5, shrunk at the ends).
`core/form_analyzer.py → median_filter()`.

## 14.11 Immerkær noise σ, Gaussian kernel, histogram equalisation, Laplacian

Formulas in sections 10.4-10.6 and 11.2.

## 14.12 FPS smoothing

Processing FPS in `AnalysisPipeline.process()`: `fps = 0.9·fps + 0.1·(1/dt)` (the first frame
uses 1/dt directly), where dt is the time `process()` took.

---

# 15. Normalization

"Normalization" appears in three concrete forms in this project.

## 15.1 Normalised landmark coordinates (from the model)

MediaPipe returns x and y in [0, 1] relative to the image width and height. The project converts
them to **pixels** immediately (`x·w`, `y·h`, section 12.5), because angles are identical in
either system and pixels are convenient for drawing.

## 15.2 Body-proportion normalisation of distances (scale invariance)

A raw pixel distance changes with how far you stand from the camera. Dividing by a body-segment
length measured in the same frame cancels the scale:

```text
wrist_above_shoulder = (shoulder_y − wrist_y) / torso_length        (torso lengths)
hip_above_knee       = (knee_y − hip_y) / thigh_length               (thigh lengths)
knee_over_toe        = (knee_x − toe_x) · facing / shin_length       (shin lengths)
hip_line_offset      = perpendicular distance / |shoulder − ankle|   (body lengths)
knee_width_ratio     = |knee_L − knee_R| / |ankle_L − ankle_R|       (ratio)
```

Example: if you step back and appear half as large, both the wrist-shoulder gap and the torso
length halve, so the ratio is unchanged. This is why thresholds like "knee over toe ≤ 0.25 shin
lengths" work for any person, any distance, any resolution.

## 15.3 Score normalisation

Every check is mapped onto the same 0-100 scale by `check_score()`, so checks measured in degrees
and checks measured in ratios can be averaged together.

**Coordinate system reminder:** origin at the **top-left** of the processed frame, x to the
right, **y downward**, units = pixels of the resized (≤ 640 px wide) frame.

---

# 16. Feature extraction

`core/feature_extractor.py → FeatureExtractor.extract()` computes **15 features** for the side of
the body facing the camera. Each feature is `None` if any landmark it needs has visibility < 0.5.
All values are then EMA-smoothed (α = 0.5).

| # | Feature | Landmarks used | Calculation | Unit | Used by |
|---|---|---|---|---|---|
| 1 | `knee_angle` | hip, knee, ankle | `calculate_angle(hip, knee, ankle)` | deg | squat rep signal, depth check; recognition (knee ROM); overlay |
| 2 | `hip_angle` | shoulder, hip, knee | `calculate_angle(sh, hip, knee)` | deg | overlay for squat and push-up; statistics |
| 3 | `ankle_angle` | knee, ankle, toe | `calculate_angle(knee, ank, toe)` | deg | statistics only |
| 4 | `elbow_angle` | shoulder, elbow, wrist | `calculate_angle(sh, el, wr)` | deg | rep signal for curl, push-up, press; their depth/height/lockout checks; recognition (elbow ROM); overlay |
| 5 | `shoulder_angle` | hip, shoulder, elbow | `calculate_angle(hip, sh, el)` | deg | overlay for curl and press; statistics |
| 6 | `torso_inclination` | hip → shoulder | `angle_from_vertical(hip, sh)` (0 = upright) | deg | squat back angle, curl torso swing, press upright torso; recognition fallback |
| 7 | `body_inclination` | ankle → shoulder | `angle_from_vertical(ank, sh)` (0 = standing, ~90 = lying) | deg | recognition (push-up orientation) |
| 8 | `upper_arm_elevation` | shoulder → elbow | `angle_from_vertical(sh, el, upward=False)` (0 = arm hanging) | deg | curl elbow stability |
| 9 | `forearm_from_vertical` | elbow → wrist | `angle_from_vertical(el, wr)` (0 = forearm vertical, pointing up) | deg | press wrist-over-elbow |
| 10 | `body_line_angle` | shoulder, hip, ankle | `calculate_angle(sh, hip, ank)` | deg | statistics only |
| 11 | `hip_line_offset` | hip vs shoulder–ankle line | `signed_distance_to_line(hip, sh, ank)` (+ = below = sag) | body lengths | push-up hip sag and hip pike |
| 12 | `wrist_above_shoulder` | shoulder, wrist, torso length | `(sh_y − wr_y) / torso_len` | torso lengths | recognition (press vs curl) |
| 13 | `hip_above_knee` | hip, knee, thigh length | `(knee_y − hip_y) / thigh_len` | thigh lengths | statistics only |
| 14 | `knee_over_toe` | knee, toe, shin length, facing | `(knee_x − toe_x) · facing / shin_len` | shin lengths | squat knee tracking (side view only) |
| 15 | `knee_width_ratio` | both knees and ankles | `abs(kneeL_x − kneeR_x) / abs(ankleL_x − ankleR_x)` | ratio | squat knee alignment (front view only) |

### How each is produced (pattern)

```text
landmarks (pixels, visibility ≥ 0.5)
        ↓
geometry function (angle / angle-from-vertical / distance ratio / line offset)
        ↓
raw value (or None)
        ↓
EMA smoothing  S_t = 0.5·X_t + 0.5·S_(t−1)
        ↓
Features.values[name]
        ↓
rep counter (rep feature), form checks, recognizer, overlay, session log
```

### Supporting decisions inside `extract()`

* **Facing direction** for `knee_over_toe`: `facing = +1` if the toe is to the right of the heel,
  `−1` if to the left (requires |toe_x − heel_x| > 1 px). Multiplying by `facing` makes "knee
  forward of toe" positive whichever way the person faces.
* **View detection:** if both shoulders are visible and
  `|left_shoulder_x − right_shoulder_x| / torso_length > 0.45`, the view is `"front"`. In front
  view, `knee_width_ratio` is computed (if the ankles are more than 5 px apart) and
  `knee_over_toe` is set to `None` (forward travel cannot be seen from the front).
* **Side selection** (`_update_side()`): smoothed mean visibility of the six main joints per side;
  switches only with a 0.15 margin. Starts on the left side.
* **No person:** `extract()` resets the EMA memory and returns an empty `Features`.

**Features that are logged but not used by any rule:** `ankle_angle`, `body_line_angle`,
`hip_above_knee`. They appear in the per-frame CSV and the joint-angle statistics.

---

# 17. Exercise recognition

## 17.1 Rule-based, not machine learning

Exercise recognition in this project is a **hand-written, rule-based classifier** with fuzzy
scores. It is **not** a trained machine-learning model: there is no training data, no learned
weights, no neural network. The rules and numbers are in
`core/exercise_recognizer.py` and `config/thresholds.py → RECOGNITION`. (The only machine-learning
component of the whole project is the pre-trained MediaPipe pose model.)

## 17.2 The three cues

The four exercises differ in three observable ways:

| Cue | Feature | Separates |
|---|---|---|
| **Body orientation** | median `body_inclination` (ankle→shoulder from vertical) over the window; falls back to `torso_inclination` if the legs are out of frame, then to 0 (upright) | push-up (lying, ~70-90°) vs everything else (standing, ~0-15°) |
| **Which joint moves** | range of motion (P95 − P5) of `knee_angle` and of `elbow_angle` | squat (knee moves) vs arm exercises (elbow moves) |
| **Hand height** | median `wrist_above_shoulder` | shoulder press (hands above shoulders) vs curl (hands below) |

## 17.3 From cues to scores

Inside `ExerciseRecognizer.classify(samples)`:

```python
horizontal  = _ramp(orientation, 50 - 15, 50 + 15)      # 0 at 35°, 1 at 65°
upright     = 1.0 - horizontal
knee_moves  = _ramp(rom("knee_angle"),  15, 40)          # 0 below 15° ROM, 1 at 40°+
elbow_moves = _ramp(rom("elbow_angle"), 15, 40)
hands_high  = _ramp(median(wrist_above_shoulder), -0.1, 0.2)

scores = {
    "pushup":         horizontal * max(elbow_moves, 0.6),   # a held plank still reads as push-up
    "squat":          upright * knee_moves,
    "shoulder_press": upright * elbow_moves * hands_high * (1 - knee_moves),
    "bicep_curl":     upright * elbow_moves * (1 - hands_high) * (1 - knee_moves),
}
```

Each score is a **product of ramps**, which behaves like a fuzzy "AND": the press score is high
only if the person is upright AND the elbow moves AND the hands are high AND the knees are not
moving. The factor `(1 - knee_moves)` stops a squat with moving arms from being called a press or
curl.

Then:

* the label with the highest score wins;
* if that score is below `min_confidence = 0.5`, the label is `"unknown"`;
* fewer than `min_samples = 8` frames in the window → `"unknown"`;
* a range of motion needs at least 8 non-missing samples, otherwise it is treated as 0.

**Real example** (end of the `squat_bad` clip): squat 0.978, bicep curl 0.008,
shoulder press 0.003, push-up 0.0 → **squat, confidence 0.98**.

## 17.4 Window and vote (stability)

`ExerciseRecognizer.update(t, features)`:

1. Appends `(t, features)` to a sliding window and drops samples older than **2.0 s**.
2. Calls `classify()` on the window.
3. Appends the raw label to a vote queue of the last **15** frames.
4. Takes the most common label. **Only if** it is a known exercise **and** holds a strict majority
   (more than half of the recent votes) does `self.current` change.
5. Returns `self.current`.

Consequences:

* **No flicker:** one odd frame cannot switch the exercise.
* **Resting keeps the exercise:** when you stand still between sets, the knee/elbow ROM drops,
  the raw label becomes "unknown", but "unknown" never replaces a known exercise. So the rep count
  and phase stay visible.
* **Recognition needs movement:** at the very start (standing still), everything is "unknown" and
  the UI says "Auto-detect: start moving to identify the exercise". In the real `squat_bad` run
  the squat was recognised at t = 2.13 s, during the first descent (knee ROM had grown past the
  ramp and the vote reached a majority).
* **When no person is detected**, the recognizer is not updated; the last result is kept.

## 17.5 Per exercise: what triggers recognition

| Exercise | Required | Landmarks |
|---|---|---|
| **Push-up** | body line near horizontal (median inclination ≳ 50°, full score at 65°); elbow movement helps but a held plank still scores 0.6 × horizontal | ankle, shoulder (or hip, shoulder), elbow, wrist |
| **Squat** | upright and knee ROM ≥ 15° (full at 40°) | hip, knee, ankle, shoulder |
| **Shoulder press** | upright, elbow ROM, hands above the shoulders (median wrist-above-shoulder ≳ 0.05 torso lengths, full at 0.2), knees still | shoulder, elbow, wrist, hip |
| **Bicep curl** | upright, elbow ROM, hands below the shoulders, knees still | shoulder, elbow, wrist, hip |

## 17.6 Ambiguity and failure

* Ambiguous windows produce similar scores; the highest wins, and the vote smooths decisions over
  15 frames.
* If no score reaches 0.5, recognition fails politely: `exercise` stays `None` (or the previously
  recognised exercise), the phase shows "-", reps are not attributed, and the warning
  "Auto-detect: start moving to identify the exercise" appears.
* **Manual selection** is the fallback: choosing an exercise in the **Exercise** control creates
  a pipeline with `manual_exercise=<key>`. Then only that exercise's tracker exists, and the
  recogniser's result is logged but **not used** to decide anything.

## 17.7 How recognition and counting interact (important)

In auto mode `AnalysisPipeline.__init__` creates **one `ExerciseTracker` for each of the four
exercises**, and `process()` updates **all of them every frame**. When a tracker finishes a rep,
the rep is stored only if that tracker's exercise is the **active** one (the current recognition).
So the squat tracker has already been following the first squat before recognition succeeds, and
the first rep is still counted when it completes. A rep completed while the exercise is still
unknown would be discarded.

---

# 18. Squat analysis

Definition: `exercises/squat.py → SQUAT`; thresholds:
`config/thresholds.py → EXERCISE_THRESHOLDS["squat"]`.

| Property | Value |
|---|---|
| Rep signal | `knee_angle` (hip-knee-ankle) |
| Starts extended? | yes |
| Start zone | knee ≥ **160°** (`extended`) |
| Turn zone | knee ≤ **125°** (`flexed`) |
| Minimum rep time | 0.6 s |
| Phase names | STANDING, DESCENDING, BOTTOM, ASCENDING |
| Angles drawn on video | knee angle, hip angle |
| Best camera view | side (depth, back angle, knee tracking); front for knee alignment |

## 18.1 The movement as states

```text
knee ≥ 160 (2 frames)      → START     "STANDING"
knee < 160                  → TO_TURN   "DESCENDING"   (movement start time recorded)
knee ≤ 125 (2 frames)      → TURN      "BOTTOM"
knee > 125                  → TO_START  "ASCENDING"
knee ≥ 160 (2 frames)      → START     "STANDING"   and, if ≥ 0.6 s since the descent began:
                                        REP + 1  → the rep is scored
```

Between 125° and 160° is the **hysteresis dead band**: the angle can jitter there without any
effect.

## 18.2 Form checks

| Check (key) | Feature | Over the rep | Ideal → 100 | Zero → 0 | Weight | Live | Feedback when score < 75 |
|---|---|---|---|---|---|---|---|
| Depth (`depth`) | `knee_angle` | **min** | 95° | 140° | 1.5 | no | "Go deeper: knee angle only reached {value} deg" |
| Back angle (`back_angle`) | `torso_inclination` | **max** | 40° | 70° | 1.0 | yes | "Keep your back more upright: torso leaned {value} deg from vertical" |
| Knee tracking (`knee_tracking`) | `knee_over_toe` | **max** | 0.25 | 0.7 | 1.0 | yes | "Knees travelled far past the toes ({value} shin lengths)" |
| Knee alignment (`knee_alignment`) | `knee_width_ratio` | **min** | 0.9 | 0.6 | 1.0 | yes | "Keep your knees aligned with your feet: knees moved inward (knee/ankle width {value})" |
| Control (tempo) (`tempo`) | rep duration | max | 1.0 s | 0.3 s | 0.5 | no | "Rep was fast ({value} s): slow down and control the movement" |

Notes:
* **Depth** uses the smallest knee angle reached: a knee angle of 95° or less scores 100; 140° or
  more scores 0.
* **Back angle** uses the largest forward lean during the rep: up to 40° is fine (some forward
  lean is normal in a squat), 70° scores 0.
* **Knee tracking** is only measurable from the side (`None` in front view).
* **Knee alignment** (knee valgus) is only measurable from the front (`None` in side view); in a
  side-view session it shows "n/a" and is excluded from the score.

## 18.3 Real example (demo clip `squat_bad`, rep 1)

| Check | Measured | Score |
|---|---|---|
| Depth | min knee angle **113.7°** | (113.7 − 140)/(95 − 140) = 0.585 → **58.5** |
| Back angle | max torso lean **61.6°** | (61.6 − 70)/(40 − 70) = 0.28 → **28.0** |
| Knee tracking | max 0.039 shin lengths | ratio 1.47 → clipped → **100** |
| Knee alignment | not measurable (side view) | **n/a** |
| Tempo | 1.53 s | ratio 1.76 → clipped → **100** |

Rep score = (1.5·58.5 + 1·28.0 + 1·100 + 0.5·100) / (1.5 + 1 + 1 + 0.5) = 265.7 / 4 = **66.4**
→ grade "Needs improvement". Issues: "Go deeper: knee angle only reached 114 deg" and "Keep your
back more upright: torso leaned 62 deg from vertical". These are exactly the numbers shown in the
app's report.

---

# 19. Bicep curl analysis

`exercises/bicep_curl.py → BICEP_CURL`; thresholds `EXERCISE_THRESHOLDS["bicep_curl"]`.

| Property | Value |
|---|---|
| Rep signal | `elbow_angle` (shoulder-elbow-wrist) |
| Starts extended? | yes (arm straight down) |
| Start zone | elbow ≥ **130°** |
| Turn zone | elbow ≤ **90°** |
| Phase names | ARM EXTENDED, CURLING, TOP, LOWERING |
| Angles drawn | elbow angle, shoulder angle |
| Best view | side |

```text
elbow ≥ 130 (2 frames)  → ARM EXTENDED
elbow < 130              → CURLING
elbow ≤ 90 (2 frames)   → TOP            (maximum contraction)
elbow > 90               → LOWERING
elbow ≥ 130 (2 frames)  → ARM EXTENDED  → REP + 1 (if ≥ 0.6 s)
```

| Check | Feature | Over rep | Ideal | Zero | Weight | Live | Meaning |
|---|---|---|---|---|---|---|---|
| Curl height | `elbow_angle` | min | 55° | 100° | 1.5 | no | how far the forearm comes up |
| Full extension | `elbow_angle` | max | 155° | 125° | 1.0 | no | the arm straightens between reps (the start-position frames are included, section 5.12) |
| Elbow stability | `upper_arm_elevation` | max | 20° | 50° | 1.0 | yes | the upper arm should stay beside the body, not swing forward |
| Torso stability | `torso_inclination` | **range** | 6° | 20° | 1.0 | no | how much the torso rocked (cheating with momentum) |
| Control (tempo) | duration | max | 1.0 s | 0.3 s | 0.5 | no | |

Validation (README table): `bicep_curl_good` average 96, `bicep_curl_bad` 48 with "curl higher",
"upper arm swing", "torso swing" flagged.

---

# 20. Push-up analysis

`exercises/pushup.py → PUSHUP`; thresholds `EXERCISE_THRESHOLDS["pushup"]`.

| Property | Value |
|---|---|
| Rep signal | `elbow_angle` |
| Starts extended? | yes (arms straight at the top) |
| Start zone | elbow ≥ **145°** |
| Turn zone | elbow ≤ **120°** |
| Phase names | TOP, LOWERING, BOTTOM, PUSHING UP |
| Angles drawn | elbow angle, hip angle |
| Best view | side (required) |

```text
elbow ≥ 145 → TOP → (elbow < 145) LOWERING → (elbow ≤ 120) BOTTOM → (elbow > 120) PUSHING UP
            → (elbow ≥ 145) TOP → REP + 1
```

**Body alignment calculation:** `hip_line_offset = signed_distance_to_line(hip, shoulder, ankle)`.
The shoulder-ankle line is where a straight body would be. The hip's perpendicular distance from
it, divided by the body length, is:
* **positive** when the hip is below the line → **sagging** hips,
* **negative** when the hip is above the line → **piking** (hips in the air).

| Check | Feature | Over rep | Ideal | Zero | Weight | Live |
|---|---|---|---|---|---|---|
| Depth | `elbow_angle` | min | 90° | 130° | 1.5 | no |
| Lockout | `elbow_angle` | max | 160° | 135° | 1.0 | no |
| Body alignment (sag) | `hip_line_offset` | max | 0.04 | 0.15 | 1.25 | yes |
| Body alignment (pike) | `hip_line_offset` | min | −0.04 | −0.15 | 1.0 | yes |
| Control (tempo) | duration | max | 1.0 s | 0.3 s | 0.5 | no |

The sag check takes the **maximum** offset (worst sag in the rep); the pike check takes the
**minimum** (most negative = worst pike). A hip within ±4 % of body length of the line scores 100
on both.

Recognition note: a push-up is recognised by the **horizontal body**, section 17.

---

# 21. Shoulder press analysis

`exercises/shoulder_press.py → SHOULDER_PRESS`; thresholds `EXERCISE_THRESHOLDS["shoulder_press"]`.

This is the only exercise that **starts flexed** (`start_extended=False`): the rep begins with the
hands at the shoulders (elbow bent) and the turn point is the lockout overhead (elbow straight).
The `RepCounter` swaps its zones accordingly.

| Property | Value |
|---|---|
| Rep signal | `elbow_angle` |
| Start zone | elbow ≤ **110°** (rack) |
| Turn zone | elbow ≥ **135°** (lockout) |
| Phase names | RACK, PRESSING, LOCKOUT, LOWERING |
| Angles drawn | elbow angle, shoulder angle |

```text
elbow ≤ 110 (2 frames) → RACK
elbow > 110            → PRESSING
elbow ≥ 135 (2 frames) → LOCKOUT
elbow < 135            → LOWERING
elbow ≤ 110 (2 frames) → RACK → REP + 1 (if ≥ 0.6 s)
```

| Check | Feature | Over rep | Ideal | Zero | Weight | Live | Meaning |
|---|---|---|---|---|---|---|---|
| Lockout | `elbow_angle` | max | 165° | 135° | 1.5 | no | press fully overhead |
| Range (bottom) | `elbow_angle` | min | 95° | 125° | 1.0 | no | lower to shoulder level |
| Upright torso (`back_arch`) | `torso_inclination` | max | 10° | 28° | 1.0 | yes | do not lean back |
| Wrist over elbow (`wrist_stack`) | `forearm_from_vertical` | max | 15° | 45° | 1.0 | yes | forearm vertical under the weight |
| Control (tempo) | duration | max | 1.0 s | 0.3 s | 0.5 | no | |

---

# 22. Repetition counting

`core/rep_counter.py → RepCounter`, used through `core/pipeline.py → ExerciseTracker`.

## 22.1 The idea

A rep is a **complete, ordered movement**: leave the start position, reach the turn position,
come back. Counting "peaks" of a noisy angle signal would double-count jitter. A state machine
with two separated thresholds (**hysteresis**) counts only real movements.

## 22.2 Zones

For an exercise that starts extended (squat, curl, push-up):

```text
angle ≥ extended            → "start" zone
angle ≤ flexed              → "turn" zone
flexed < angle < extended   → no zone (dead band)
```

For the shoulder press (starts flexed) the zone names are swapped
(`RepCounter._zone()`).

## 22.3 States and transitions

```text
                ┌──────────────────────────────────────────────────────────┐
                │                                                          │
   WAITING ──(start zone confirmed 2 frames)──► START                      │
                                                  │                        │
                                   (angle leaves start zone)               │
                                   record move_start = t                   │
                                                  ▼                        │
                                               TO_TURN ──(start zone confirmed, never turned)──► START
                                                  │          → "partial" event if the extreme
                                  (turn zone confirmed 2 frames)   was ≥ 15° past the start threshold
                                                  ▼                        │
                                                TURN                       │
                                                  │                        │
                                     (angle leaves turn zone)              │
                                                  ▼                        │
                                              TO_START ──(turn zone confirmed)──► TURN (bounce)
                                                  │
                                   (start zone confirmed 2 frames)
                                                  ▼
                                               START  ── if t − move_start ≥ 0.6 s:
                                                          count += 1, emit "rep" event
Any state: joint missing for more than 1.5 s → reset() → WAITING (in-progress rep discarded)
```

| State | Phase shown (squat) | Meaning |
|---|---|---|
| `WAITING` | GET IN POSITION | not yet seen in the start position |
| `START` | STANDING | resting in the start position |
| `TO_TURN` | DESCENDING | moving towards the turn point |
| `TURN` | BOTTOM | in the turn zone |
| `TO_START` | ASCENDING | returning |

## 22.4 Noise handling (four layers)

1. **EMA smoothing** of the angle before it reaches the counter (section 32).
2. **Hysteresis dead band** between `flexed` and `extended` (35° for the squat).
3. **Debouncing:** entering a zone must be **confirmed on 2 consecutive frames**
   (`confirm_frames=2`, tracked by `_streak_zone`/`_streak`). Leaving a zone is immediate.
4. **Minimum rep duration** 0.6 s: a "rep" faster than that (e.g. a glitch) is not counted.

Plus: **missing-landmark timeout** (`max_missing_seconds = 1.5`): short gaps are tolerated, a long
absence resets the machine so two halves of different movements are never stitched together.

## 22.5 Why hysteresis prevents duplicate reps

With a single threshold at, say, 140°, an angle wobbling 139-141° at the bottom would cross it
many times and could count several reps. With two thresholds, after reaching the turn zone
(≤ 125°) the machine only counts again after the angle has gone all the way back above 160°. A
wobble inside the 125-160° band changes nothing. The TO_START → TURN "bounce" transition also means
dipping back down during the ascent does not start a new rep.

## 22.6 Partial reps

If you start descending but come back up **without reaching the turn zone**, and you went at least
15° past the start threshold (`partial_min_deg`), a `"partial"` event is produced. The pipeline
scores it (so you still get feedback), stores it in `SessionLog.partials`, **does not count it**,
shows "Partial rep not counted: bottom position was not reached" and sets the status to bad.
Smaller dips (< 15°) are ignored completely.

## 22.7 Frame-by-frame (real trace, `squat_bad`, 30 fps, first rep)

| t (s) | knee angle (smoothed) | internal squat state | phase shown | reps |
|---|---|---|---|---|
| 0.00 | none (first frame) | WAITING | - (no exercise yet) | 0 |
| 0.20 | 174.5 | START (confirmed ≥ 160) | - | 0 |
| 1.40 | 173.4 | START | - | 0 |
| ~1.70 | < 160 | **TO_TURN**, move_start = 1.70 | - | 0 |
| 2.00 | 128.6 | TO_TURN | - | 0 |
| 2.13 | 124.0 | **TURN** (≤ 125 for 2 frames) | **BOTTOM** (squat just recognised) | 0 |
| 2.40 | 116.2 | TURN (minimum ≈ 113.7 in this rep) | BOTTOM | 0 |
| 2.77 | 128.4 | **TO_START** | ASCENDING | 0 |
| 3.00 | 139.8 | TO_START | ASCENDING | 0 |
| 3.23 | 163.4 | **START** (≥ 160 for 2 frames), 1.53 s ≥ 0.6 s | **STANDING** | **1** |

The rep's time span is 1.70 → 3.23 s (duration 1.53 s), exactly what the report shows. The phase
column is "-" until recognition succeeds, even though the squat tracker was already running (see
17.7).

---

# 23. Form analysis

`core/form_analyzer.py → FormAnalyzer`.

## 23.1 Per-rep evaluation (`evaluate_rep`)

For each check in `exercise.all_checks` (the exercise's checks + tempo):

```text
Rule (FormCheck)
  ↓
Measurement:
  tempo:  value = end_time − start_time
  others: series = [frame[feature] for frames in the rep if not None]
          value  = reduce( median_filter(series) )      reduce ∈ {min, max, range}
  ↓
Threshold: (ideal, zero) from EXERCISE_THRESHOLDS
  ↓
Score: check_score(value, ideal, zero)  → 0..100   (None if value is None)
  ↓
Condition: score < 75 (STATUS_GOOD)
  ↓
Feedback: check.message.format(value=value)  appended to issues
```

Then `result.score = weighted_score((score, weight) for each check)`.

**Which frames:** the tracker passes `start_frames` (up to 60 frames resting in the start
position before the movement) + `rep_frames` (the movement) + the final frame. Including the
resting frames matters for checks like curl "full extension": the arm is straightest *before*
the curl starts.

**Why the median filter:** a single mis-detected frame (a landmark spike) could otherwise set the
min or max. The 5-sample running median removes one- or two-frame spikes while keeping the true
plateau at the bottom of a rep (a mean would shave it).

## 23.2 Live posture checks (`live`)

Every frame, for checks with `live=True`, the current feature value is scored immediately. These
are posture rules that make sense on a single frame (back angle, knee tracking, knee alignment,
elbow stability, hip sag/pike, torso upright, wrist stack). Range-of-motion rules (depth,
lockout, curl height, tempo) need a whole rep, so they are not live.

## 23.3 From checks to on-screen feedback (`AnalysisPipeline._compose_feedback`)

1. Live checks with status ≠ good → their messages (with the current value) are the first
   feedback lines.
2. If the most recent event was a **partial** rep: "Partial rep not counted: <turn phase>
   position was not reached" + that partial's issues.
3. Otherwise the **last counted rep's** issues are added.
4. **Status** = the worst of: each live check's status, the last rep's score status, and "bad"
   if the last event was a partial. Rank: neutral < good < warning < bad.

The status colours the skeleton, the frame border and the read-out panel.

## 23.4 Every rule in one table

| Exercise | Rule | Measurement | Threshold (ideal → zero) | Feedback trigger |
|---|---|---|---|---|
| Squat | Depth | min knee angle | 95° → 140° | score < 75 (knee min > ~106°) |
| Squat | Back angle | max torso inclination | 40° → 70° | > ~47.5° |
| Squat | Knee tracking | max knee-over-toe | 0.25 → 0.7 | > ~0.36 |
| Squat | Knee alignment | min knee/ankle width | 0.9 → 0.6 | < ~0.83 |
| Curl | Curl height | min elbow angle | 55° → 100° | > ~66° |
| Curl | Full extension | max elbow angle | 155° → 125° | < ~147.5° |
| Curl | Elbow stability | max upper-arm elevation | 20° → 50° | > ~27.5° |
| Curl | Torso stability | range of torso inclination | 6° → 20° | > ~9.5° |
| Push-up | Depth | min elbow angle | 90° → 130° | > 100° |
| Push-up | Lockout | max elbow angle | 160° → 135° | < ~153.75° |
| Push-up | Sag | max hip offset | 0.04 → 0.15 | > ~0.0675 |
| Push-up | Pike | min hip offset | −0.04 → −0.15 | < ~−0.0675 |
| Press | Lockout | max elbow angle | 165° → 135° | < 157.5° |
| Press | Bottom range | min elbow angle | 95° → 125° | > 102.5° |
| Press | Upright torso | max torso inclination | 10° → 28° | > 14.5° |
| Press | Wrist over elbow | max forearm from vertical | 15° → 45° | > 22.5° |
| All | Tempo | rep duration | 1.0 s → 0.3 s | < 0.825 s |

(The trigger column is where the score crosses 75: value = zero + 0.75 × (ideal − zero).)

---

# 24. Scoring system

`core/scoring.py`. There are **no penalties subtracted from 100**; every check is scored on its
own 0-100 scale and the rep score is a **weighted average**. Nothing else is added.

## 24.1 Check score

```python
def check_score(value, ideal, zero):
    ratio = (value - zero) / (ideal - zero)
    return max(0.0, min(1.0, ratio)) * 100.0
```

* **Input:** measured value, ideal (score 100), zero (score 0).
* **Limits:** clipped to [0, 100]. Better than ideal is still 100; worse than zero is still 0.
* Raises `ValueError` if ideal == zero.

## 24.2 Rep score

```python
def weighted_score(items):
    pairs = [(s, w) for s, w in items if s is not None]
    total_w = sum(w for _, w in pairs)
    if not pairs or total_w == 0:
        return None
    return sum(s * w for s, w in pairs) / total_w
```

```text
Rep score = Σ(wᵢ · sᵢ) / Σ wᵢ     over the checks that could be measured
```

**Weights** (from the exercise files): the main range-of-motion check has 1.5 (squat depth, curl
height, push-up depth, press lockout); push-up sag 1.25; other checks 1.0; tempo 0.5.

**Unmeasurable checks** (value `None`, e.g. knee alignment in side view) are **excluded** from
both the numerator and the denominator, so they neither help nor hurt the score. They appear as
"n/a".

## 24.3 Session score

`SessionLog.summary()`: `average_score` = plain mean of the scores of all counted reps
(partial reps are excluded); best and worst rep by score.

## 24.4 Grade bands (`grade()`)

Rounded score compared top-down with `SCORE_BANDS`:

| Score | Grade |
|---|---|
| 100 | Excellent detected form |
| 90-99 | Very good |
| 75-89 | Acceptable |
| 60-74 | Needs improvement |
| 0-59 | Poor detected form |
| `None` | Not enough data |

## 24.5 Status colours (`status()`)

| Score | Status | Colour |
|---|---|---|
| ≥ 75 | good | green |
| 50-74 | warning | amber |
| < 50 | bad | red |
| `None` | neutral | grey |

The same 75 threshold decides whether a check produces a feedback message (section 23).

## 24.6 Worked example

See 18.3: squat rep with depth 58.5 (w 1.5), back angle 28.0 (w 1), knee tracking 100 (w 1),
knee alignment n/a, tempo 100 (w 0.5) → 265.7 / 4.0 = **66.4 → "Needs improvement"**, status
**warning** (amber).

---

# 25. Visualization

## 25.1 On the video frame (OpenCV), `processing/visualization.py`

`draw_overlay(img, st)` is called at the end of every `AnalysisPipeline.process()`:

1. **Skeleton** (`draw_skeleton()`), if a pose was detected:
   * line thickness `lw = max(2, frame_width // 250)`;
   * a line for each `SKELETON` connection whose two landmarks both have visibility ≥ 0.5,
     `cv2.line(..., color, lw, cv2.LINE_AA)` (anti-aliased), in the **status colour**;
   * a circle on every visible body landmark (index > 10, so no face points): filled white
     (`COLORS["joint"]`) with a 1-px ring in the status colour, radius `lw + 2`
     (`cv2.circle`).
2. **Angle labels** (`draw_angles()`), if an exercise is active: for each of the exercise's
   `display_features` (e.g. knee and hip angle for the squat), the text "`<value> deg`" is drawn
   10 px right and above the vertex joint of the analysed side (`FEATURE_VERTEX` maps
   `knee_angle → knee`).
3. **Info panel** (top-left): a dark rectangle (width 62 % of the frame) blended at 60 % opacity
   (`cv2.addWeighted`) with the lines: "`<Exercise>  [AUTO 97%]`" or "`[MANUAL]`",
   "`Reps: n   Phase: <phase>`", "`Last rep score: s (<grade>)`" in the score's colour, and
   "`FPS: x`".
4. **Messages** (bottom-left): up to 2 warnings (amber), then up to 3 feedback messages in the
   status colour, or "GOOD FORM" in green if a rep exists and there is no feedback. Truncated to 80
   characters.
5. **Status border:** a 4-px rectangle around the whole frame in the status colour.

Text is drawn by `_text()` with `cv2.putText` (font `FONT_HERSHEY_SIMPLEX`), plus four copies
shifted by one pixel in black to form an outline, so it is readable on any background. Font scale
grows with the frame width (`max(0.4, w / 1100)`).

**Colours** (`config/thresholds.py → COLORS`, **BGR** order): good (80, 200, 80) green,
warning (0, 200, 255) amber, bad (60, 60, 230) red, neutral (230, 230, 230), joint white, panel
(30, 30, 30).

## 25.2 In the Streamlit page

* The annotated frame is shown with `st.image(annotated, channels="BGR", width="stretch")`
  (camera) or through the `on_frame` callback in `run_video_analysis()` (files, every 3rd frame).
* The read-out panel is HTML built by `ui/components.py → status_panel()`: exercise name, mode,
  status tag (GOOD FORM / CHECK FORM / CORRECT YOUR FORM / WAITING), REPS, PHASE, LAST REP,
  live checks with PASS / WATCH / FIX tags (good / warning / bad), last-rep breakdown, feedback,
  warnings, telemetry.
* Results page charts: Altair (`ui/charts.py → ui_score_per_rep`, `ui_angle_vs_time`,
  `ui_rep_durations`) and the HTML form-breakdown bars.
* PDF charts: Matplotlib (`angle_vs_time`, `score_per_rep`, `rep_durations`,
  `check_breakdown`).
* Illustrations: SVG from `ui/figures.py` (section 7.2).

---

# 26. Video processing

`processing/video_processor.py`.

## 26.1 VideoCapture and frame reading

* `video_info(path)`: opens the file with `cv2.VideoCapture`, fails with a friendly
  `VideoSourceError` if it cannot open or read a first frame; returns FPS (30 if missing or
  invalid) and frame count (`CAP_PROP_FRAME_COUNT`).
* `iterate_video(path, frame_skip)`: a Python **generator** that reads frames one at a time with
  `cap.read()` and yields `(frame, idx / fps, idx)` for every `frame_skip`-th frame. The capture
  is released in `finally:`.

## 26.2 "Analyse every n-th video frame" (frame skipping)

* **What:** the slider (1-4, default 1) in the Processing menu, only shown for files. `n = 2`
  analyses frames 0, 2, 4, …
* **Implementation:** `if idx % max(1, frame_skip) == 0: yield …` (every frame is still decoded,
  but only every n-th is analysed).
* **Timestamps stay correct**, because they come from the original frame index.
* **Trade-off:**
  * higher processing frequency (n = 1): smoothest signals, most accurate min/max per rep, best
    for fast movements, slowest;
  * lower frequency (n = 2-4): about n× faster, but fewer samples per rep. The rep counter needs 2
    confirming frames per zone, and the recognizer needs 8 samples per 2-second window, so very
    fast reps or high n can make them less reliable.
* The output video is written at `fps / frame_skip`, so it plays at real speed.

## 26.3 Processing resolution

"Processing width (px)" (320 / 480 / 640 / 960, default 640) is `max_width` for
`resize_frame()`. Smaller = faster inference and preprocessing, slightly less precise landmarks.

## 26.4 FPS

* **Processing FPS** (`FrameStatus.fps`): time spent inside `process()`, smoothed.
* **Display FPS** (camera only): time between displayed frames, smoothed; includes camera and
  browser overhead.
* Real run in this environment: the `squat_bad` clip (495 frames, 640×480) was analysed in 16.9 s,
  about **29 frames per second end to end** including decoding. On the live camera with nobody in
  view, processing ran at about 73 FPS (no landmarks to track).

## 26.5 Output video

`create_writer(stem, fps, size)` tries WebM/VP8 (`"VP80"`, playable in browsers) and falls back to
MP4 (`"mp4v"`). The file is `outputs/<video stem>_analyzed.webm`. If no encoder works, the
analysis still runs and just does not save a video. The annotated frames are the **processed**
frames (resized, possibly enhanced), with the overlay.

## 26.6 `analyze_video()`

```text
video_info → for each (frame, t, idx) in iterate_video:
    annotated, status = pipeline.process(frame, t)
    first time: create the writer with the annotated size
    writer.write(annotated)
    on_frame(annotated, status, progress)       ← UI preview + progress bar
release the writer
if no frame was analysed: raise VideoSourceError
return the output path
```

## 26.7 Memory considerations

* Frames are **streamed**: only the current frame is in memory, never the whole video.
* `SessionLog.frames` grows by one small dict per analysed frame (about 30 values). A 2-minute
  webcam session at 30 FPS is about 3,600 dicts, which is small.
* The UI preview is throttled to every 3rd frame (`run_video_analysis()`), because sending images
  to the browser is the slowest part.
* Uploaded files are held by Streamlit and written to `outputs/uploads/` (upload limit 500 MB in
  `.streamlit/config.toml`).
* The rep buffers are bounded: `_start_frames` keeps at most 60 frames.

---

# 27. Demo mode

## 27.1 What it is

Ten short videos of a **rendered articulated figure** exercising, generated by
`demo/synthetic_athlete.py`. They let you demonstrate and test the app without a camera, and
because the movement is scripted, the correct answer (exercise, number of reps) is known.

## 27.2 How the clips are made

* **Forward kinematics:** each pose is built from segment angles. `u(phi)` is the unit vector at
  angle `phi` from vertical-up. Starting from the ankle, `_standing()` adds segment lengths along
  those directions: shin 95 px, thigh 100, torso 135, upper arm 75, forearm 70, head 48 above the
  shoulder, plus heel and toe points.
* **Scripted motion:** `Scenario.progress(t)` gives the movement progress p ∈ [0, 1] with a smooth
  cosine profile `p = (1 − cos 2πτ) / 2` over `rep_seconds = 2.4` s, followed by a 0.5 s pause,
  after a 1.2 s lead-in. `squat_pose(p, params)`, `curl_pose()`, `press_pose()`, `pushup_pose()`
  interpolate joint angles between the start and turn positions using parameters such as
  `knee_min`, `lean_max`, `elbow_min`, `sag`.
* **Rendering:** `draw_figure()` paints thick coloured limbs (far-side limbs darker and offset),
  a head with hair, dumbbells for curl/press; `_background()` is a grey gradient with a floor; a
  little Gaussian sensor noise is added (σ 2, or σ 7 in low light). The low-light scenario scales
  brightness to 28 %; the two-people scenario adds a smaller standing bystander.
* **Output:** 640×480, 30 FPS, VP8 WebM, via `render_scenario()`. `ensure_samples()` renders any
  missing clip into `data/sample/` (on first use; the clips are not stored in Git).

## 27.3 The ten scenarios (`SCENARIOS`)

| Clip | Exercise | Reps | Scripted difference |
|---|---|---|---|
| squat_good | squat | 5 | full depth, upright torso |
| squat_bad | squat | 5 | shallow (knee min 112°), forward lean 62° |
| bicep_curl_good | curl | 5 | full range, still upper arm |
| bicep_curl_bad | curl | 5 | half curls, arm swing 45°, torso sway 16° |
| pushup_good | push-up | 5 | deep, straight body |
| pushup_bad | push-up | 5 | shallow, hips sagging |
| shoulder_press_good | press | 5 | full lockout, stacked wrists |
| shoulder_press_bad | press | 5 | partial lockout, leaning back, wrists forward |
| squat_low_light | squat | 4 | dark and noisy (tests CLAHE + denoising) |
| squat_two_people | squat | 3 | a bystander in frame (tests multi-person handling) |

## 27.4 Does it use the real pipeline?

**Yes.** Demo analysis calls exactly the same `run_video_analysis()` → `analyze_video()` →
`AnalysisPipeline.process()` as an uploaded file. MediaPipe has to find the rendered figure;
nothing is read from the script during analysis. Every count and score is **measured**. The
"Scripted ground truth" panel only displays what the script intended, so you can compare.

## 27.5 Differences from webcam mode

| | Demo | Webcam |
|---|---|---|
| Source | WebM file | camera |
| Timestamps | frame index ÷ FPS | wall clock |
| Mirror | no | optional |
| Frame skip | available | not applicable |
| Output video | saved to `outputs/` | not saved |
| Ground truth | known | unknown |
| Session length | clip length | up to the time limit or Stop |

## 27.6 Batch validation

**Run batch validation** analyses all ten clips in auto mode with a fresh
`AnalysisPipeline(PoseDetector(), …)` each (no output video) and builds a table: expected vs
detected exercise, expected vs counted reps, average score, and a `correct` flag (true if the only
exercise with counted reps is the expected one **and** the rep count matches). The README records
10/10 correct.

---

# 28. Session state

`st.session_state` is a dictionary that survives Streamlit reruns for one browser session (one
tab). Refreshing the page or opening a new tab starts a new, empty session. Here is every key the
project uses.

### `st.session_state["page"]`
* **Purpose:** which page to draw (`Home`, `Live Analysis`, `Video Upload`, `Demo Mode`,
  `Exercise Analysis`, `Results`, `About`).
* **Created:** `app.py` with `setdefault`, from the `?page=` URL parameter if valid, else `"Home"`.
* **Modified:** `ui/pages.py → goto()` (every navigation button, the source switcher, "Open full
  report", the empty-state buttons on Results).
* **Read by:** `app.py` (dispatch), `ui/pages.py → top_nav()` (active link).

### `st.session_state["webcam_running"]`
* **Purpose:** whether the camera loop should run.
* **Created:** implicitly; `live()` reads it with `.get(..., False)`.
* **Modified:** **Start** (`_set("webcam_running", True)`), **Stop** (`False`), set to `False`
  when the camera cannot be opened and when the loop ends.
* **Read by:** `ui/pages.py → live()`.

### `st.session_state["last_session"]`
* **Purpose:** the most recent `SessionLog` (the data behind every report).
* **Created/modified:** `live()` stores `pipeline.session` **at the start** of a camera session
  (a live reference, so it is filled while running and survives Stop);
  `ui/components.py → run_video_analysis()` stores it after a file/demo analysis.
* **Read by:** `live()` ("Last camera session"), `_analysis_result_block()` (shown only if its
  `source` matches the current file/clip), `results()`.

### `st.session_state["last_output"]`
* **Purpose:** path of the annotated output video.
* **Modified:** `run_video_analysis()` (path or `None`); `live()` sets it to `None` (camera
  sessions are not recorded).
* **Read by:** `_analysis_result_block()` (the "Annotated output video" expander).

### `st.session_state["batch"]`
* **Purpose:** the pandas DataFrame of batch-validation results.
* **Modified:** `demo()` after **Run batch validation**. **Read by:** `demo()`.

### `st.session_state["pdf"]`
* **Purpose:** bytes of the last built PDF report.
* **Modified:** `results()` when **Build PDF report** is clicked. **Read by:** `results()` (shows
  **Download report (PDF)**).
* **Note:** it is not cleared when a new analysis finishes, so after analysing something new,
  click **Build PDF report** again before downloading.

### Settings keys (`read_settings()` defaults, `settings_bar()` widgets)

| Key | Widget | Default | Settings field |
|---|---|---|---|
| `exercise_choice` | Exercise segmented control | `"auto"` | `manual_exercise` (`None` if auto) |
| `max_width` | Processing width select slider | 640 | `max_width` |
| `frame_skip` | Analyse every n-th video frame | 1 | `frame_skip` |
| `enhance` | Adaptive enhancement checkbox | `True` | `enhance` |
| `focus` | Background focus checkbox | `False` | `background_focus` |
| `cam` | Camera index | 0 | `webcam_index` |
| `cam_secs` | Max session length (s) | 120 | `webcam_seconds` |
| `mirror` | Mirror view | `True` | `mirror` |

* **Created:** `read_settings()` on every run (re-assigned so they persist when hidden).
* **Modified:** the widgets on the Analyze pages; `_keep_a_choice()` resets `exercise_choice` to
  `"auto"` if you click the already-selected segment (which would otherwise clear it).
* **Read by:** `read_settings()` → the `Settings` object passed to every page.

### Other keys
* `source_choice`: Analyze source switcher; set by `_analyze_header()` before drawing, changed by
  the user, handled by `_source_changed()`.
* `guide_ex`: selected exercise on the Exercises page.
* `dip_t_upload`: time slider of the DIP pipeline viewer.
* Button keys (`nav_home`, `nav_analyze`, `nav_exercises`, `nav_results`, `nav_about`,
  `nav_cta`, `cta_start`, `cta_demo`, `home_guide`, `live_report`, `report_<source>`): Streamlit
  stores widget state under these; the project uses them for CSS targeting and tests.

### URL query parameter
`?page=<page name>`: read once at session start (`app.py`), written by `goto()`.

**Persistence:** nothing is saved to disk except the uploaded file and the annotated video in
`outputs/`. A session's results exist only in memory until you download CSV/PDF.

---

# 29. Data flow

| Stage | Input | Processing | Output | File |
|---|---|---|---|---|
| UI routing | click / URL | `goto()` sets `page`; script reruns | page function called | `app.py`, `ui/pages.py` |
| Settings | widgets | `read_settings()` | `Settings` | `ui/components.py` |
| Session setup | `Settings`, source label | `make_pipeline()` → `PoseDetector`, `AnalysisPipeline` | fresh pipeline | `ui/components.py`, `core/pipeline.py` |
| Camera acquisition | camera index | `open_webcam()`, `cap.read()`, optional flip | BGR frame, wall-clock t | `processing/video_processor.py`, `ui/pages.py → live()` |
| File acquisition | path, frame skip | `video_info()`, `iterate_video()` | BGR frame, t = idx/fps | `processing/video_processor.py` |
| Preprocessing | BGR frame, previous ROI | resize, stats, noise, Gaussian?, CLAHE?, →RGB | `PreprocessResult` | `processing/preprocessing.py` |
| Pose detection | RGB, ms | MediaPipe VIDEO mode, ×w/h, prominence | `PoseDetection` | `core/pose_detector.py` |
| Subject processing | landmarks, BGR | ROI, Laplacian variance, mask?, background? | ROI, sharpness, display frame | `processing/image_processing.py` |
| Features | `PoseDetection` | side/view, geometry, EMA | `Features` (15 values) | `core/feature_extractor.py`, `core/angle_calculator.py` |
| Recognition | features, t | window, ramps, vote | `Recognition` | `core/exercise_recognizer.py` |
| Rep counting | rep angle, t | hysteresis state machine ×(1 or 4) | state/phase, `RepEvent` | `core/rep_counter.py`, `core/pipeline.py` |
| Form analysis | rep frames | median, min/max/range, `check_score`, messages | `RepResult` | `core/form_analyzer.py` |
| Live checks | current features | live checks only | `LiveCheck` list | `core/form_analyzer.py` |
| Scoring | values, weights | clip-linear, weighted mean, bands | scores, grade, status | `core/scoring.py` |
| Feedback | live checks, last rep/partial | messages, worst status | `FrameStatus` fields | `core/pipeline.py` |
| Warnings | pose, preprocessing, features | rules | `FrameStatus.warnings` | `core/pipeline.py` |
| Overlay | frame, `FrameStatus` | OpenCV drawing | annotated BGR | `processing/visualization.py` |
| Logging | status, features, preprocessing | append dict | `SessionLog.frames/reps/partials` | `core/pipeline.py`, `core/session.py` |
| Live display | annotated frame, status | `st.image`, `status_panel()` | web page | `ui/pages.py`, `ui/components.py` |
| Output video | annotated frames | VP8 writer | `outputs/*_analyzed.webm` | `processing/video_processor.py` |
| Report | `SessionLog` | summary, breakdown, statistics, charts | Results page | `core/session.py`, `ui/pages.py`, `ui/charts.py` |
| Export | `SessionLog` | CSV table; PDF text + charts | downloads | `utils/export.py` |

---

# 30. Error handling

| Situation | What happens | Where |
|---|---|---|
| **No camera / camera busy / no permission** | `open_webcam()` tries DirectShow then any backend and requires one successful `read()`; otherwise raises `VideoSourceError("Camera N is unavailable. Check …")`. `live()` catches it, shows `st.error`, sets `webcam_running = False`. | `processing/video_processor.py → open_webcam()`, `ui/pages.py → live()`; test `test_missing_camera_shows_error_not_crash` |
| **Camera stops mid-session** | more than 30 consecutive failed reads → "The camera stopped delivering frames (disconnected or in use)." and the loop ends; `finally:` releases camera and detector. | `live()` |
| **Wrong file type** | extension checked by `is_supported_video()` → `st.error("Unsupported file type…")`. The uploader also restricts types. | `utils/helpers.py`, `upload()` |
| **Corrupt / unreadable video** | `video_info()` raises `VideoSourceError` ("Could not open the video - the file may be corrupt…", or "contains no readable frames"). Caught in `upload()` and in `run_video_analysis()`. | `processing/video_processor.py`, `ui/pages.py`, `ui/components.py` |
| **Video with no analysable frames** | `analyze_video()` raises `VideoSourceError("No frames could be analysed…")`. | `analyze_video()` |
| **No person in the frame** | `PoseDetection()` with `detected = False`; features reset to empty; the rep counter receives `None`; warning "No person detected - step into the camera view" (or "Person left the frame - tracking paused" if someone was seen before). | `core/pose_detector.py`, `core/feature_extractor.py`, `core/pipeline.py → _collect_warnings()` |
| **Missing landmarks** (visibility < 0.5) | each affected feature is `None`, never a guessed value; the skeleton omits those joints; if the exercise's rep feature is missing: warning "Key joints not visible - make sure the whole body is in view"; the counter tolerates gaps up to 1.5 s, then resets to WAITING. | `FeatureExtractor.extract() → p(), safe()`, `RepCounter.update()` |
| **Degenerate geometry** (two landmarks at the same point) | `calculate_angle` / `signed_distance_to_line` raise `ValueError`; `safe()` turns it into `None`. | `core/angle_calculator.py`, `core/feature_extractor.py` |
| **Low confidence** | MediaPipe drops poses below its 0.5 thresholds; recognition below 0.5 confidence stays "unknown" with the warning "Auto-detect: start moving to identify the exercise"; unmeasurable checks show "n/a" and are excluded from the score. | `core/pose_detector.py`, `core/exercise_recognizer.py`, `core/scoring.py` |
| **Multiple people** | the most prominent person is analysed; warning "N people detected - analysing the most prominent person only". | `core/pose_detector.py → prominence()`, `_collect_warnings()` |
| **Invalid exercise key** | `AnalysisPipeline.__init__` raises `ValueError("Unknown exercise …")`. The UI can only offer valid keys, so this protects programmatic use. | `core/pipeline.py` |
| **Pose model crash on a frame** | `detect()` catches any exception and returns `PoseDetection(error=…)`; the error text becomes a warning; the session continues. | `core/pose_detector.py` |
| **Pose model missing** | `ensure_pose_model()` downloads it; if the download fails, `RuntimeError` with manual instructions; `app.py` catches `RuntimeError` around the page and shows it with `st.error`. `run_video_analysis()` also catches `RuntimeError`. | `utils/helpers.py`, `app.py`, `ui/components.py` |
| **Non-increasing timestamps** | forced to increase in `detect()` (VIDEO mode would otherwise throw). | `core/pose_detector.py` |
| **Poor lighting** | measured; CLAHE (and denoising) applied if enabled; warning "Poor lighting - contrast enhancement (CLAHE) applied" or "Poor lighting detected". | `processing/preprocessing.py`, `_collect_warnings()` |
| **Blurry frame** | Laplacian variance in the ROI < 40 → warning "Frame is blurry - landmarks may be less reliable". | `processing/image_processing.py → sharpness()`, `_collect_warnings()` |
| **No video encoder** | `create_writer()` returns `None`, logs a warning; analysis continues without an output video. | `processing/video_processor.py` |
| **Low detection rate** | if a person was found in < 60 % of frames, the report shows a warning. | `ui/components.py → session_overview()`, `ui/pages.py → results()` |
| **Empty frame passed to preprocessing** | `preprocess_frame()` raises `ValueError("Empty frame")`. Frame sources only yield successfully read frames, so this is a guard for programmatic use; it is not specially caught in the UI. | `processing/preprocessing.py` |

---

# 31. Performance

The target is real-time analysis on a laptop **CPU**. The decisions:

| Decision | Effect | Where |
|---|---|---|
| **Lite** pose model (5.8 MB, float16) | fastest BlazePose variant, adequate accuracy for large body joints | `config/thresholds.py → POSE_MODEL_URL` |
| **VIDEO running mode** | tracking between frames; the person detector does not run on every frame | `PoseDetector.__init__` |
| **Resize to ≤ 640 px** (INTER_AREA) | fewer pixels for every OpenCV operation and for the model; anti-aliased | `resize_frame()` |
| **Adaptive enhancement** | CLAHE and Gaussian cost time; they run only when measured as needed | `preprocess_frame()` |
| **Noise estimate on a 3×3 filter, statistics with `cv2.meanStdDev`** | vectorised C++ operations, negligible cost | `preprocessing.py` |
| **Segmentation off by default** | the mask costs extra inference; only on with "Background focus" | `make_pipeline()` |
| **One detector per session, closed afterwards** | model loaded once per session, native memory freed | `ui/components.py`, `ui/pages.py` |
| **Frame skip for files** | analyse every n-th frame when speed matters | `iterate_video()` |
| **UI throttling** | preview updated every 3rd frame for files | `run_video_analysis()` |
| **Streaming frames** | constant memory | `iterate_video()` |
| **Cheap rules instead of a second model** | recognition, counting and scoring are a few arithmetic operations per frame | `core/` |
| **Smoothing** | EMA (one multiply-add per feature), running median only when a rep ends | `feature_extractor.py`, `form_analyzer.py` |
| **Cached illustrations** | `lru_cache` on the skeleton samples | `ui/figures.py → rep_cycle()` |

**Measured here:** the whole `squat_bad` clip (495 frames, 640×480) ran at about **29 FPS end to
end** including decoding and logging; the README reports about 40 FPS processing on 640×480 video
and about 30 FPS end to end with the live webcam in the browser on the development laptop.
Numbers depend on the CPU.

---

# 32. Smoothing and noise reduction (on signals)

Landmarks jitter by a few pixels from frame to frame even when you stand still. Three filters work
on the **feature time series** (not on images).

## 32.1 Exponential moving average (EMA), every feature, every frame

`core/feature_extractor.py → FeatureExtractor._smooth()`:

```text
S_t = α · X_t + (1 − α) · S_(t−1)        α = FEATURES["ema_alpha"] = 0.5
```

* **X_t:** the raw feature value this frame (e.g. knee angle from the landmarks).
* **S_t:** the smoothed value passed on; **S_(t−1)** the previous smoothed value.
* **α:** how much the newest measurement counts. α = 1 means no smoothing; small α means strong
  smoothing but more lag.
* **Start:** the first value after a gap is used as is (`S = X`).
* **Missing value:** clears the memory for that feature (so stale values never leak into a new
  appearance of the joint).

**Example** (α = 0.5): previous S = 160°, jitter frame X = 150° → S = 155°. A one-frame glitch
X = 100° → S = 130° (halved), and the next normal frame X = 160° → S = 145°, recovering quickly.

**Effect of α:**

| α | Behaviour |
|---|---|
| 1.0 | raw signal, all jitter passes through |
| 0.5 (used) | jitter halved; lag about one frame (≈ 33 ms at 30 FPS) |
| 0.2 | very smooth but lags several frames: rep bottoms would be detected late and the minimum angle underestimated |

**Why it is needed:** without smoothing, jitter near a threshold could flip the rep counter's zone
and make recognition noisy.

## 32.2 Running median, per rep

`core/form_analyzer.py → median_filter()`, window 5, applied to each check's series before taking
min/max/range. The median removes **impulse noise** (one wrong frame) completely, which a mean or
EMA only reduces. Example: series [118, 117, 90, 116, 118] → min 90 without the filter, but the
median at the middle position is 117, so the reported depth is about 116, not the glitch 90.

## 32.3 Other smoothing

* **Side-visibility smoothing:** `0.7·old + 0.3·new` and a 0.15 switching margin
  (`_update_side()`).
* **Recognition:** a 2-second window, percentile ROM (P5-P95), median orientation, a 15-frame
  majority vote.
* **Rep counter:** 2-frame confirmation and hysteresis (section 22).
* **FPS displays:** `0.9·old + 0.1·new`.
* **Images:** the Gaussian blur (section 10.5) is the only image-domain noise filter.

---

# 33. Complete user example: one squat

This example uses the real numbers from analysing the demo clip `squat_bad` in **auto** mode
(30 FPS video, so one frame ≈ 33 ms). A webcam session works the same way, except `t` comes from
the clock.

**Before anything moves (t = 0 - 1.4 s)**

```text
Frame at t = 0.00
 ↓ preprocess_frame: 640×480 (no resize), brightness ≈ 184, contrast OK,
   noise σ ≈ 0.08 → nothing applied
 ↓ PoseDetector.detect → 1 person, 33 landmarks
 ↓ FeatureExtractor: first frame, knee angle not yet available
 ↓ Recognizer: < 8 samples → unknown
 ↓ Trackers (all 4): WAITING
 ↓ Screen: skeleton in grey (neutral), "Detecting... [AUTO]", warning
   "Auto-detect: start moving to identify the exercise"

Frames t = 0.07 - 1.40
 ↓ knee ≈ 174-175° (standing), torso ≈ 3°
 ↓ Squat tracker: 2 frames ≥ 160° → START (internally "STANDING")
 ↓ Recognizer: knee ROM ≈ 0 → squat score 0 → still unknown
 ↓ Phase shown "-" because no exercise is active yet
```

**Descending (t ≈ 1.4 - 2.1 s)**

```text
t = 1.60  knee 168.5°, torso 13.8°
t ≈ 1.70  knee drops below 160° → squat tracker START → TO_TURN, move_start = 1.70
t = 1.80  knee 147.7°, torso 30.2°   (in the 125-160° dead band)
t = 2.00  knee 128.6°, torso 45.8°
          live check "Back angle": torso 45.8° → score (45.8−70)/(40−70) = 0.81 → 81 (good)
```

**Bottom (t ≈ 2.1 - 2.75 s)**

```text
t = 2.13  knee 124.0° (≤ 125° for 2 frames) → TURN
          Recognizer: knee ROM in the 2-s window is now large, upright, elbows still
          → squat score ≈ 1.0 → majority of the last 15 votes → exercise = SQUAT
          Screen: "Squat [AUTO 100%]", phase BOTTOM, reps 0
          live check back angle: torso 54° → score 53 → WATCH (amber);
          feedback "Keep your back more upright: torso leaned 54 deg from vertical"
t = 2.40  knee 116.2°, torso 61.3° → back-angle live score 29 → FIX (red), border red
          (the median-filtered minimum over the rep will be 113.7°)
```

**Ascending (t ≈ 2.77 - 3.2 s)**

```text
t = 2.77  knee 128.4° (> 125°) → TURN → TO_START   phase ASCENDING
t = 3.00  knee 139.8°
t = 3.20  knee 162.2° (first frame ≥ 160°)
```

**Rep complete (t = 3.23 s)**

```text
t = 3.23  knee 163.4° (second frame ≥ 160°) → confirmed START
          duration 3.23 − 1.70 = 1.53 s ≥ 0.6 s → count = 1, RepEvent("rep", 1.70, 3.23)
 ↓ ExerciseTracker.update: frames = resting start frames + rep frames + this frame
 ↓ FormAnalyzer.evaluate_rep:
     depth          min(median(knee))   = 113.7° → 58.5
     back_angle     max(median(torso))  = 61.6°  → 28.0
     knee_tracking  max(knee_over_toe)  = 0.039  → 100
     knee_alignment side view            = n/a    → excluded
     tempo          1.53 s                        → 100
     rep score = (1.5·58.5 + 28.0 + 100 + 0.5·100) / 4.0 = 66.4
     issues: "Go deeper: knee angle only reached 114 deg",
             "Keep your back more upright: torso leaned 62 deg from vertical"
 ↓ pipeline: squat is the active exercise → RepResult index 1 appended to SessionLog.reps
 ↓ Screen: REPS 1, PHASE STANDING, LAST REP 66/100 ("Needs improvement"),
   "Last rep" breakdown: Depth 58 FIX, Back angle 28 FIX, Knee tracking 100 PASS,
   Knee alignment n/a, Control (tempo) 100 PASS; feedback lists both issues;
   overall status warning (amber) because the last rep scored 66.
```

**After five reps** the report shows: total reps 5, average 65.8-66 ("Needs improvement"), best
rep #1 (66), worst #2 (65), duration 0:16, detection rate 100 %, form breakdown Depth ≈ 57,
Back angle ≈ 28, Knee tracking 100, Knee alignment n/a, Control 100.

---

# 34. What happens when you click each button

The original UI had buttons named "Start Webcam Analysis", "Upload Video", "Demo Mode",
"Exercise Analysis", "View Results" and "About Project". The redesigned UI uses the names below;
the mapping is given for each.

### START ANALYSIS (nav bar `nav_cta`, Home `cta_start`), formerly "Start Webcam Analysis"
1. `on_click=goto("Live Analysis")` → `page = "Live Analysis"`, URL `?page=Live Analysis`.
2. The script reruns; `live()` draws the Camera page in standby. **The camera is not opened yet.**

### START (Camera page)
1. `_set("webcam_running", True)`; rerun.
2. `live()` sees `running = True`: `open_webcam(settings.webcam_index)` (error → message, stop).
3. `make_pipeline(settings, "webcam 0")`: creates a `PoseDetector` (loads the model; segmentation
   only if Background focus) and an `AnalysisPipeline` (manual or auto).
4. `last_session = pipeline.session`, `last_output = None`.
5. Loop until the time limit: read → mirror → `pipeline.process(frame, t)` → show annotated frame
   and `status_panel()`.
6. At the end: release camera and detector, `webcam_running = False`, `st.rerun()` → standby with
   "Last camera session".

### STOP (Camera page)
`_set("webcam_running", False)` → Streamlit interrupts the running loop at its next UI call; the
`finally:` block releases the camera and closes the detector; the rerun shows standby and the
"Last camera session" summary. Data recorded so far stays in `last_session`.

### ANALYZE → VIDEO FILE (formerly "Upload Video"), then ANALYSE VIDEO
1. Source switcher (or nav) → `page = "Video Upload"`.
2. Choosing a file: saved to `outputs/uploads/<name>`, type checked, `video_info()` validates it and
   shows frames/FPS/duration.
3. **Analyse video** returns `True` in that rerun → `run_video_analysis(path, settings,
   "upload: <name>")`: new pipeline → `analyze_video()` with the frame skip → live preview every
   3rd frame, progress bar → writes `outputs/<name>_analyzed.webm` → stores `last_session`,
   `last_output`.
4. `_analysis_result_block()` shows the headline numbers, the output video and **Open full
   report**.

### TRY A DEMO CLIP / ANALYZE → DEMO CLIPS (formerly "Demo Mode"), then ANALYSE THIS CLIP
1. `page = "Demo Mode"`; `ensure_samples([name])` renders the clip if missing (spinner); the clip
   plays in the video player; the ground-truth panel is shown.
2. **Analyse this clip** → `run_video_analysis(path, settings, "demo: <name>")` (same path as an
   upload).
3. Result block as above.

### RUN BATCH VALIDATION
Renders any missing clips, analyses all ten in auto mode (no output videos), builds the table,
stores it in `st.session_state["batch"]`, and shows "N / 10 clips: exercise and rep count match
the script."

### EXERCISE (segmented control on Analyze pages)
Changes `exercise_choice`. **Auto detect** → next analysis uses automatic recognition; a lift →
next analysis is manual (`AnalysisPipeline(manual_exercise=key)`). It does not affect an analysis
already running.

### PROCESSING (popover)
Changes `max_width`, `frame_skip`, `enhance`, `focus`, `cam`, `cam_secs`, `mirror` for the **next**
analysis.

### EXERCISES (nav), formerly "Exercise Analysis"
`page = "Exercise Analysis"` → `exercise_guide()` draws the definitions and illustrations. No
analysis, no model.

### RESULTS (nav) / OPEN FULL REPORT, formerly "View Results"
`page = "Results"` → `results()` reads `last_session` and builds the report; or shows the empty
state if there is none.

### DOWNLOAD REPORT (CSV) / DOWNLOAD REPETITIONS (CSV) / DOWNLOAD PER-FRAME ANGLES (CSV)
The CSV text is generated during the rerun (`report_csv()`, `reps_df().to_csv()`,
`frames_df().to_csv()`) and the browser downloads it.

### BUILD PDF REPORT → DOWNLOAD REPORT (PDF)
`report_pdf(session)` renders text pages and Matplotlib charts into a multi-page PDF in memory,
stores the bytes in `st.session_state["pdf"]`, and the download button appears.

### ABOUT (nav), formerly "About Project"
`page = "About"` → static content.

### HOME (nav)
`page = "Home"`.

### HOW EACH LIFT IS SCORED →
`page = "Exercise Analysis"`.

---

# 35. DIP concepts mapped to the project

Only techniques that are implemented are listed.

| DIP concept | Where it is used | File → function |
|---|---|---|
| Image acquisition | webcam capture and video decoding | `video_processor.open_webcam()`, `iterate_video()` |
| Digital image representation / quantisation | frames are 8-bit (0-255) BGR arrays; the mask is quantised to {0, 255} | throughout; `binary_person_mask()` |
| Temporal sampling | a video is a sequence of frames sampled at the file's FPS; frame skipping subsamples it | `iterate_video()` |
| Spatial resampling (geometric transformation) | downscaling with area interpolation (anti-aliasing) | `preprocessing.resize_frame()` |
| Colour spaces | BGR, RGB, greyscale (luminance), LAB | `preprocess_frame()`, `apply_clahe()` |
| Image statistics / histogram analysis | mean (brightness), standard deviation (contrast) on the ROI; histogram in the DIP viewer | `grey_statistics()`, `components._histogram()` |
| Noise estimation | Immerkær Laplacian-difference estimator | `estimate_noise_sigma()` |
| Spatial filtering (convolution) | `filter2D` with the noise mask; Gaussian blur | `estimate_noise_sigma()`, `preprocess_frame()`, `suppress_background()` |
| Low-pass filtering / noise reduction | Gaussian 5×5 denoising | `preprocess_frame()` |
| Contrast enhancement / histogram equalisation | CLAHE on the L channel | `apply_clahe()` |
| Edge detection (second derivative) | Laplacian; variance as a sharpness/blur measure; edge map | `laplacian_edges()`, `sharpness()`, `edge_map_for_display()` |
| Thresholding / segmentation | person probability → binary mask | `binary_person_mask()` |
| Morphological processing | opening and closing with an elliptical structuring element | `binary_person_mask()` |
| Region of interest | subject bounding box from landmarks; statistics measured inside it | `landmarks_roi()`, `preprocess_frame()` |
| Image blending / compositing | alpha blending foreground and blurred background; translucent info panel | `suppress_background()`, `draw_overlay()` (`addWeighted`) |
| Min-max normalisation | edge map scaled to 0-255 for display | `edge_map_for_display()` |
| Feature extraction | landmarks → joint angles and normalised distances | `feature_extractor.extract()` |
| Image analysis / pattern recognition | rule-based exercise recognition, rep state machine, form checks | `exercise_recognizer`, `rep_counter`, `form_analyzer` |
| Signal smoothing (1-D filtering) | EMA and running median on feature sequences | `_smooth()`, `median_filter()` |
| Graphics overlay | lines, circles, text drawn into the image | `visualization.py` |

**Not implemented** (do not claim these): Fourier/frequency-domain filtering, Canny or Sobel edge
detection, Hough transform, image restoration (deconvolution), compression, colour
quantisation/clustering, optical flow.

---

# 36. Why each technology was chosen

### Python
All the needed libraries (OpenCV, MediaPipe, NumPy, pandas, Streamlit) have first-class Python
APIs, so one language covers image processing, the model, the maths, the UI and the tests. Readable
code matters for a project that must be explained.

### OpenCV
In this project OpenCV reads webcam and video frames (`VideoCapture`), resizes them
(`resize` with `INTER_AREA`), converts colour spaces (`cvtColor`), measures statistics
(`meanStdDev`), filters (`filter2D`, `GaussianBlur`, `Laplacian`), enhances contrast
(`createCLAHE`), does morphology (`morphologyEx`, `getStructuringElement`), draws the overlay
(`line`, `circle`, `putText`, `rectangle`, `addWeighted`) and writes the annotated video
(`VideoWriter`). It is fast (C++), standard for DIP, and works directly on NumPy arrays.

### MediaPipe
It provides a pre-trained, CPU-real-time pose model with 33 landmarks including feet, per-landmark
visibility, a VIDEO tracking mode and an optional segmentation mask, installable with pip. Training
a pose model is far beyond the scope of a DIP project.

### NumPy
Landmarks are (33, 2) arrays; visibility filtering, bounding boxes, percentiles, medians, the noise
sum and mask arithmetic are all vectorised NumPy operations. OpenCV images *are* NumPy arrays.

### Streamlit
A complete web UI (buttons, file upload, live image updates, charts, downloads, session state) in
pure Python, without writing a separate front end and API. Ideal for a real-time demo on
`localhost`.

### pandas
The session log becomes DataFrames for the tables, statistics (`angle_statistics`,
`check_breakdown`, `issue_counts`) and one-line CSV export.

### Matplotlib
Generates the PDF report pages and charts without an extra PDF library (`PdfPages`), and the
histogram in the DIP viewer. The non-interactive `Agg` backend works headless.

### Altair
Interactive charts (hover tooltips) on the Results page; it comes with Streamlit, so no extra
dependency.

### pytest
86 automated tests from unit maths to full end-to-end runs through the real model and the UI.

---

# 37. Alternatives

| Choice | Alternative | Why the current choice |
|---|---|---|
| **MediaPipe BlazePose** | OpenPose | OpenPose is heavier (needs a GPU for real time), harder to install, and gives 25 keypoints; BlazePose runs on a CPU and gives 33 with visibility. |
| | MoveNet / PoseNet (TensorFlow) | MoveNet is fast but gives 17 keypoints (no heel/toe, needed for knee tracking). |
| | YOLO-pose / HRNet | accurate, but heavier and usually GPU-bound. |
| **OpenCV** | PIL / Pillow | Pillow handles loading and simple edits but has no video capture, CLAHE, Laplacian, morphology or video writing. |
| | scikit-image | good DIP algorithms but slower and no camera/video I/O. |
| **Streamlit** | Flask / Django + JavaScript | full control but needs a separate front end, templates, and streaming frames manually. |
| | Tkinter / PyQt desktop app | no browser, harder to style and chart; Streamlit gives charts and downloads for free. |
| **Rule-based recognition** | ML classifier (SVM, random forest, LSTM/TCN on landmark sequences) | needs a labelled dataset and training; harder to explain; rules are transparent, need no data, and reach 10/10 on the demo clips. An LSTM/TCN is listed as future work. |
| **Hysteresis state machine** | peak detection on the angle signal | peak detection double-counts jitter and cannot express phases or partial reps. |
| **Linear, weighted scoring** | a learned quality score | a learned score needs expert-labelled reps; the linear rule is transparent ("100 at 95°, 0 at 140°"). |
| **EMA smoothing** | Kalman filter, One-Euro filter | better for fast motion but more parameters; EMA is enough at 30 FPS. |
| **CLAHE** | global histogram equalisation, gamma correction | global equalisation over-amplifies bright backgrounds and noise; CLAHE is local and contrast-limited. |
| **Gaussian denoise** | median, bilateral, non-local means | Gaussian is cheapest and suits Gaussian sensor noise; bilateral/NLM preserve edges better but cost much more per frame. |
| **Laplacian sharpness** | Tenengrad (Sobel energy), FFT high-frequency energy | variance of the Laplacian is a standard, one-line blur measure. |

---

# 38. Limitations

These come from the implementation, not from a generic list.

* **2-D angles depend on the viewpoint.** All angles are measured in the image plane (`z` and world
  landmarks are not used). A squat filmed at 45° gives different angles than from the side. Side
  view is recommended; knee alignment needs the front view.
* **One body side.** Only the side facing the camera is analysed, so left/right asymmetry is not
  detected.
* **One person.** With several people, only the most prominent one is analysed.
* **Occlusion and clothing.** Landmarks with visibility < 0.5 are dropped; baggy clothes, dumbbells
  in front of joints or partial framing reduce what can be measured (checks show "n/a").
* **Model accuracy.** The lite model is less precise than larger models; very deep elbow flexion
  tends to be underestimated.
* **Lighting.** CLAHE + denoising help a lot, but in the experiment frames at 6 % brightness were
  still undetectable.
* **Rule-based recognition over four exercises.** Other movements, unusual variants, or very small
  ranges of motion may be unknown or mislabelled; manual selection is the fallback. Recognition
  needs some movement before it decides.
* **Reps before recognition.** In auto mode a rep that *finishes* before the exercise is recognised
  is not recorded (normally recognition happens during the first rep).
* **Thresholds are defaults, not clinically validated,** and the same for every body type. The score
  measures geometry, not safety or injury risk.
* **Fixed minimum rep time** (0.6 s) and tempo rule; very fast athletes could be penalised.
* **Demo clips are rendered figures.** They validate the pipeline end to end, but they are not real
  footage; test with your own videos too.
* **No persistence.** Sessions live in memory; history is lost on refresh unless exported.
* **Webcam sessions are not recorded** as video (only files produce an annotated output).
* **CPU-bound frame rate**, which varies by laptop; at low FPS the 2-frame confirmation and
  recognition window react more slowly.

---

# 39. Future improvements

(None of these are implemented.)

* **3-D pose:** use MediaPipe's world landmarks (metres, hip-centred) for viewpoint-independent
  angles.
* **Both sides and symmetry:** analyse left and right together and report asymmetry.
* **Learned recognition:** a temporal model (LSTM, TCN or transformer) over landmark sequences, for
  more exercises and variants.
* **Personalised calibration:** record a user's range of motion and set thresholds per person.
* **More exercises:** lunges, deadlifts, rows; each is a new `exercises/*.py` file plus thresholds.
* **Better filtering:** One-Euro or Kalman filters for fast movements.
* **Voice feedback:** spoken cues ("go deeper") during the set.
* **Workout history:** store sessions (e.g. SQLite) and show progress over time.
* **Record webcam sessions** as annotated video.
* **Mobile/edge deployment:** MediaPipe also runs on Android/iOS and the web.

---

# 40. Common viva questions (50)

Answers are written the way you can say them.

**1. What does your project do?**
It analyses exercise form from a webcam or a video. It finds the body's landmarks with pose
estimation, computes joint angles, recognises the exercise, counts reps with a state machine,
scores every rep against measurable rules and shows feedback live and in a report.

**2. Why did you choose OpenCV?**
Because it does everything image-related we need in one fast library: reading the webcam and video
files, resizing, colour conversion, Gaussian blur, CLAHE, the Laplacian, morphology, drawing the
skeleton and writing the output video. Its images are NumPy arrays, so they plug straight into the
rest of the code.

**3. What is MediaPipe?**
Google's library of pre-trained on-device vision models. We use its Pose Landmarker with the
BlazePose lite model. We give it an RGB frame and it returns 33 body landmarks with a visibility
score each.

**4. What is pose estimation?**
Finding the positions of a person's body joints, like shoulders, elbows, hips and knees, in an
image. The output is a set of keypoints, not a label.

**5. What are landmarks?**
The keypoints the model returns: 33 named points, for example 25 is the left knee. Each has an x
and y position and a visibility value. We convert x and y from 0-to-1 fractions into pixels.

**6. How do you calculate joint angles?**
With vectors. For the knee I take hip minus knee and ankle minus knee, compute the dot product,
divide by both lengths and take the arc cosine. That gives the angle at the knee in degrees, from
0 to 180. It's `calculate_angle` in `core/angle_calculator.py`.

**7. Why use angles instead of raw coordinates?**
Coordinates change when you move around the frame, stand closer or use another resolution. The
knee angle doesn't. So one threshold, like "the knee should reach 95 degrees", works for anyone.

**8. How do you count repetitions?**
With a hysteresis state machine on one joint angle, for example the knee for squats. It has to see
the start position, then the turn position, then the start position again, in that order, and the
rep must take at least 0.6 seconds. Only then is it counted.

**9. What is hysteresis?**
Using two different thresholds for "entering" and "leaving" a state. For the squat, standing is
above 160 degrees and the bottom is below 125. The gap between them is a dead band, so jitter
around one value can't toggle the state and create extra reps.

**10. Why is smoothing required?**
Landmarks jitter a few pixels every frame even when you stand still. We smooth every feature with
an exponential moving average, and we take a running median before measuring a rep's minimum or
maximum, so one bad frame can't count a rep or ruin a score.

**11. What happens when a landmark is missing?**
If its visibility is below 0.5 we treat it as missing. Every feature that needs it becomes "none"
instead of a wrong number. The skeleton skips it, the checks that need it show "n/a" and are left
out of the score, and if it's gone for more than 1.5 seconds the rep counter resets.

**12. How does squat detection work?**
The recogniser looks at a two-second window. If the body is upright and the knee angle's range of
motion is large (15 degrees starts it, 40 degrees is full), the squat score goes up to 1. It needs
a score of at least 0.5 and a majority of the last 15 frames' votes.

**13. How does bicep curl detection work?**
Upright body, elbow moving, knees not moving, and the wrists below the shoulders. If the wrists
were above the shoulders it would be a shoulder press instead.

**14. How do you tell a push-up apart?**
By the body orientation: the angle of the ankle-to-shoulder line from vertical. Standing is near 0
degrees; a push-up is near horizontal. Above about 50-65 degrees the push-up score takes over.

**15. How is incorrect form detected?**
Each exercise has a list of checks. A check takes one feature over the rep, for example the minimum
knee angle for squat depth or the maximum torso lean for back angle, and scores it between an ideal
value and a zero value. If a check scores below 75, its feedback message is shown, with the
measured number in it.

**16. How is the score calculated?**
Each check is `(value − zero) / (ideal − zero)`, clipped to 0-1, times 100. The rep score is the
weighted average of the checks that could be measured. Depth-type checks have weight 1.5, tempo
0.5, most others 1. There are no hidden penalties.

**17. Is this machine learning?**
Only the pose model is machine learning, and we use it pre-trained. Recognition, counting, form
analysis and scoring are rule-based: hand-written geometry with thresholds in one config file.
That's deliberate, so every decision can be explained.

**18. What DIP techniques are used?**
Resizing with area interpolation, colour-space conversion (BGR, RGB, grey, LAB), grey-level
statistics for lighting, Immerkær noise estimation, Gaussian filtering, CLAHE, the Laplacian for
sharpness, thresholding to a binary mask, morphological opening and closing, a region of interest,
alpha blending, and signal smoothing on the features.

**19. What is image preprocessing here?**
Everything done to a frame before the model sees it: resize, measure brightness, contrast and
noise on the person, then denoise and apply CLAHE only if the measurements say it's needed, then
convert to RGB.

**20. Why resize frames?**
Speed. A 1080p frame has almost seven times the pixels of 640 wide, and the model doesn't need them.
We use area interpolation because it averages pixels while shrinking, which avoids aliasing.

**21. What is RGB vs BGR?**
The order of the colour channels. OpenCV stores pixels as blue, green, red. MediaPipe expects red,
green, blue. So we convert just before the model with `cvtColor(BGR2RGB)`.

**22. What is Gaussian filtering?**
A blur where each pixel becomes a weighted average of its neighbours, with weights from a Gaussian
bell curve. It removes high-frequency noise. We use a 5×5 kernel, and only when the estimated noise
is high.

**23. What is CLAHE?**
Contrast-Limited Adaptive Histogram Equalisation. It equalises the histogram in small tiles, 8 by
8, so each region gets its own contrast stretch, and it clips the histogram at 2.5 so noise isn't
amplified too much. We apply it only to the lightness channel in LAB so colours don't change, and
only when the person is too dark or low-contrast.

**24. Why denoise before CLAHE?**
Because CLAHE amplifies whatever is in the image, including noise. Our experiment showed that on
dark and noisy frames, CLAHE alone found the person in 0 % of frames, but denoise then CLAHE found
them in about 90 %.

**25. How do you estimate noise?**
With Immerkær's method: convolve the grey image with a 3×3 mask that is a difference of two
Laplacians. That cancels image structure and leaves mostly noise. Then a formula turns the average
absolute response into a sigma.

**26. What is edge detection used for?**
To measure sharpness. We take the variance of the Laplacian inside the person's box. If it's below
40 the frame is blurry and we warn that landmarks may be less reliable. We don't use Canny.

**27. What is segmentation in your project?**
When Background focus is on, MediaPipe gives a person probability map. We threshold it at 0.5 to
get a binary mask, clean it with morphological opening and closing, and use it to blur and darken
the background. It's optional.

**28. What is the region of interest?**
The bounding box of the visible landmarks, padded by 15 %. We use it to measure lighting on the
person instead of the whole frame, and to measure sharpness.

**29. What is feature extraction?**
Turning 33 landmarks into meaningful numbers: joint angles (knee, hip, elbow, shoulder, ankle),
inclinations (torso, body, upper arm, forearm) and distances divided by body segment lengths
(knee over toe, hip offset, wrist height, knee width ratio). There are 15 features.

**30. What is normalisation in your project?**
Two things. The model's coordinates are normalised 0-1, and we convert them to pixels. And distances
are divided by a body segment length, like the torso or shin, so they don't depend on how big you
appear.

**31. How do you handle multiple people?**
The model can find up to two. We pick the most prominent one, which is the one with the largest
bounding box times mean visibility, analyse only them, and show a warning.

**32. What happens in bad lighting?**
We measure brightness and contrast on the person. Below 70 brightness or 30 contrast we apply CLAHE,
after denoising if needed, and show a "poor lighting" warning.

**33. How do you avoid counting half reps?**
A rep must reach the turn zone. If you go down and come back without reaching it, and moved at
least 15 degrees, it's recorded as a partial rep: you get feedback, but it isn't counted.

**34. How do you avoid counting jitter as a rep?**
Four things: EMA smoothing, the hysteresis dead band, two-frame confirmation for entering a zone,
and a minimum rep time of 0.6 seconds.

**35. What is the EMA formula?**
S_t = alpha × X_t + (1 − alpha) × S_(t−1), with alpha = 0.5. The new measurement counts half, the
history counts half.

**36. What is the running median for?**
Before taking the minimum or maximum over a rep, a five-sample running median removes one-frame
spikes, which a minimum would otherwise pick up.

**37. How are the phases shown, like DESCENDING or BOTTOM?**
They are just names for the state machine's states. For the squat: START is STANDING, TO_TURN is
DESCENDING, TURN is BOTTOM, TO_START is ASCENDING.

**38. Why does the shoulder press work differently?**
It starts with the elbow bent at the shoulders and the turn point is the straight-arm lockout, so
its start zone is below 110 degrees and its turn zone above 135. The state machine swaps the zones.

**39. How do you check push-up body alignment?**
We compute the hip's perpendicular distance from the shoulder-ankle line, divided by body length.
Positive means sagging, negative means piking. Within 4 % of body length scores 100.

**40. What is a live check?**
A posture rule that can be judged on a single frame, like back angle or hip sag. It's evaluated
every frame so the skeleton changes colour in real time. Range rules like depth need the whole rep.

**41. Where are the thresholds?**
All of them are in `config/thresholds.py`, for example squat depth ideal 95 degrees, zero 140. The
exercise files only reference them by name.

**42. How do you know the counts are correct?**
With the demo clips. They're rendered from a script, so we know the exercise and rep count. Running
the real pipeline on all ten clips gives 10 out of 10 correct exercises and rep counts. The test
suite also runs the model end to end.

**43. Are the demo results fake?**
No. The clips are synthetic, but the analysis isn't. MediaPipe has to find the rendered figure and
every count and score is measured by the same pipeline as a real video.

**44. How many tests are there and what do they cover?**
86 tests: angle maths, the rep counter with noise and partial reps, scoring, exercise definitions
and recognition, preprocessing, full end-to-end runs through MediaPipe on rendered clips, and the
Streamlit UI with AppTest.

**45. Why Streamlit?**
It gives a web UI with buttons, file upload, live image updates, charts and downloads in pure
Python, so we didn't need a separate front end. It runs locally on localhost.

**46. How does the UI talk to the processing code?**
The UI creates an `AnalysisPipeline`, feeds it frames, and gets back an annotated frame and a
`FrameStatus` object each time, which it displays. Everything is also logged into a `SessionLog`
that the Results page reads. The UI never calculates results itself.

**47. What happens when you press Start?**
A flag is set, the page reruns, the webcam is opened, a pose detector and pipeline are created,
and a loop reads frames, runs `pipeline.process()` and shows the result until you press Stop or the
time limit is reached. The camera is released in a `finally` block.

**48. How fast is it?**
About 29 frames per second end to end on a 640×480 demo clip on this laptop, CPU only. Speed comes
from the lite model, VIDEO tracking mode, resizing to 640 wide, and only enhancing when needed.

**49. What are the limitations?**
2-D angles depend on camera angle, so side view is best. One person and one body side at a time.
Occlusion and baggy clothes hide landmarks. Recognition is rules over four exercises. Thresholds
aren't clinically validated.

**50. How could you improve it?**
Use MediaPipe's 3-D world landmarks for viewpoint-independent angles, analyse both sides for
symmetry, train a sequence model for more exercises, calibrate thresholds per person, add voice
feedback, and save workout history.

---

# 41. 30-second explanation

> "Our project is a gym form analyzer that works with any webcam. Each frame is preprocessed with
> DIP techniques: resizing, and CLAHE and Gaussian denoising only when the lighting or noise
> measurements say it's needed. MediaPipe then finds 33 body landmarks. From those we compute joint
> angles, recognise the exercise with geometric rules, count reps with a hysteresis state machine,
> and score each rep out of 100 against clear rules like squat depth or back angle. The feedback
> appears live on the video and in a downloadable report."

---

# 42. 1-minute explanation

> "People training alone rarely get feedback on technique, so we built a camera-based form
> analyzer for four exercises: squats, bicep curls, push-ups and shoulder presses.
>
> Each frame goes through a pipeline. First, image preprocessing: we resize to 640 pixels wide and
> measure brightness, contrast and noise on the person. If it's dark we apply CLAHE on the lightness
> channel, and if it's noisy we apply a Gaussian blur first. Second, MediaPipe's pose model gives
> 33 landmarks. Third, we turn them into features: joint angles using the dot-product formula, and
> distances divided by body-segment lengths so they don't depend on camera distance.
>
> A rule-based recogniser identifies the exercise from body orientation, which joint moves, and
> hand height. A hysteresis state machine counts reps only when the full movement happens. Each rep
> is scored with transparent rules, and anything below 75 produces a feedback message. Everything
> is drawn on the video in real time and summarised in a report with charts and CSV or PDF export."

---

# 43. 3-minute explanation (presentation)

> **Problem.** "When you train alone, nobody tells you that your squat is too shallow or that
> you're swinging during curls. Coaches and motion-capture systems are expensive. We wanted an
> ordinary webcam to give that feedback."
>
> **Input.** "The app, built in Python with a Streamlit interface, accepts a live webcam, an
> uploaded video, or one of ten demo clips. Frames are read with OpenCV."
>
> **Image processing.** "Each frame is first resized to at most 640 pixels wide using area
> interpolation, which is fast and avoids aliasing. Then we measure it: the mean and standard
> deviation of grey levels on the person's bounding box tell us brightness and contrast, and
> Immerkær's method estimates the noise level. Only if needed, we denoise with a 5 by 5 Gaussian
> filter and then apply CLAHE on the L channel of LAB, so contrast improves without changing
> colours. The order matters: in our experiment, on dark and noisy frames, CLAHE alone detected the
> person in 0 % of frames, while denoising first raised it to about 90 %. We also compute the
> variance of the Laplacian to warn about blurry frames, and optionally threshold the segmentation
> mask and clean it with morphology to blur the background."
>
> **Pose estimation.** "The preprocessed RGB frame goes to MediaPipe's BlazePose lite model in video
> mode, which tracks the person between frames. It returns 33 landmarks with a visibility score;
> anything below 0.5 is treated as missing. If two people are visible we analyse the more
> prominent one."
>
> **Features.** "From the landmarks we compute 15 features: joint angles with the dot-product
> formula, inclinations from the vertical, and scale-invariant ratios like knee-over-toe in shin
> lengths. Each is smoothed with an exponential moving average."
>
> **Recognition and counting.** "A rule-based recogniser looks at a two-second window. Is the body
> horizontal? Does the knee or the elbow move? Are the hands above the shoulders? These give fuzzy
> scores, and a majority vote over 15 frames keeps the label stable. Reps are counted by a
> hysteresis state machine on one angle, for example the knee: above 160 degrees is standing, below
> 125 is the bottom, and a rep is only counted after start, bottom, start, in order, taking at least
> 0.6 seconds. Half reps are reported but not counted."
>
> **Form and score.** "Each exercise has declarative checks. For a squat: depth, back angle, knee
> tracking, knee alignment and tempo. Each is scored linearly between an ideal and a zero value, the
> rep score is the weighted average, and any check below 75 produces a message with the measured
> value, like 'Go deeper: knee angle only reached 114 degrees'."
>
> **Output and results.** "The skeleton, angles, rep count, phase and feedback are drawn on the
> video in real time, and the Results page shows the reps, average score, a per-check breakdown and
> charts, with CSV and PDF export. On our ten scripted demo clips it recognised every exercise and
> counted every rep correctly, and 86 automated tests cover the whole pipeline. The main limitation
> is that angles are 2-D, so side-on filming works best, and 3-D landmarks are our next step."

---

# 44. Presentation walkthrough (what to demonstrate)

Before class: run `streamlit run app.py` once so the pose model is downloaded and the demo clips
are rendered (open the Demo clips page once, or run `python -m demo.synthetic_athlete`). Close
other apps that use the camera.

1. **Open the app → Home.** Point at the headline and the animated squat skeleton: "these
   figures are generated from our own kinematics and state machine; the knee angle and phase are
   real computations."
2. **Exercises page.** Choose *Squat*: show the start and bottom positions with their angles (175°
   and 88°), the start/turn zones (≥ 160°, ≤ 125°), the phases, and the form-checks table with
   ideal/zero values and weights. This sets up everything you'll show next.
3. **Analyze → Demo clips → `squat_bad` → Analyse this clip.** While it runs, point at:
   * the skeleton and the knee/hip angle labels on the video;
   * the phase changing STANDING → DESCENDING → BOTTOM → ASCENDING and REPS incrementing;
   * the posture checks turning WATCH/FIX and the feedback "Keep your back more upright";
   * the last-rep score of about 66 and the breakdown (depth ≈ 58, back angle ≈ 28).
4. **Open full report.** Show total reps 5, average ≈ 66, the form breakdown (depth and back angle
   low), score per rep, and the knee angle over time with the shaded reps and the 160/125 threshold
   lines. Say: "this chart *is* the state machine: each shaded band is one counted rep between the
   two thresholds."
5. **Export tab.** Download the CSV; build and download the PDF.
6. **Compare with `squat_good`** (optional): average ≈ 100, no issues.
7. **Live camera.** Analyze → Camera, Exercise = *Squat* (or Auto detect), **Start**. Stand side-on
   2-3 m away, do 3 squats: show the skeleton, the knee angle, the rep counter, the colour change
   when you lean forward, then **Stop** and open the report.
8. **DIP evidence.** Analyze → Video file, upload any clip (for example a demo clip from
   `data/sample/`), open **DIP pipeline viewer**, and walk through the stages: resize, grey, CLAHE,
   Gaussian, Laplacian edges, binary mask, skeleton + ROI, histogram and the decisions table.
9. **Robustness (optional):** demo clip `squat_low_light` (CLAHE applied, still 4/4 reps) and
   `squat_two_people` (warning, correct 3 reps).
10. **Batch validation** (optional, takes a few minutes): 10/10 clips correct.
11. **About page** to close: pipeline, DIP concepts, limitations and future work.

Backup plan if the camera fails: everything in steps 3-9 works without a camera.

---

# 45. Code references index

Search for these names to jump to the code.

| Topic | File → name |
|---|---|
| Entry point, routing | `app.py → PAGE_FUNCS` |
| Page switching | `ui/pages.py → goto()`, `top_nav()` |
| Settings | `ui/components.py → read_settings()`, `settings_bar()`, `Settings`, `SETTING_DEFAULTS` |
| Create a session | `ui/components.py → make_pipeline()` |
| Camera loop | `ui/pages.py → live()` |
| File analysis with preview | `ui/components.py → run_video_analysis()` |
| Live read-out | `ui/components.py → status_panel()` |
| Report | `ui/pages.py → results()`, `_breakdown()`; `ui/components.py → kpi_strip()`, `session_overview()` |
| UI charts | `ui/charts.py → ui_score_per_rep()`, `ui_angle_vs_time()`, `ui_rep_durations()` |
| PDF charts | `ui/charts.py → score_per_rep()`, `rep_durations()`, `angle_vs_time()`, `check_breakdown()` |
| Illustrations | `ui/figures.py → rep_cycle()`, `hero_svg()`, `exercise_svg()`, `pose_pair_svg()` |
| DIP viewer | `ui/components.py → dip_stage_viewer()` |
| Webcam | `processing/video_processor.py → open_webcam()` |
| Video reading | `processing/video_processor.py → video_info()`, `iterate_video()`, `analyze_video()` |
| Output video | `processing/video_processor.py → create_writer()` |
| Whole per-frame pipeline | `core/pipeline.py → AnalysisPipeline.process()` |
| Preprocessing | `processing/preprocessing.py → preprocess_frame()` |
| Resize | `processing/preprocessing.py → resize_frame()` |
| Brightness/contrast | `processing/preprocessing.py → grey_statistics()` |
| Noise σ | `processing/preprocessing.py → estimate_noise_sigma()` |
| CLAHE | `processing/preprocessing.py → apply_clahe()` |
| ROI | `processing/image_processing.py → landmarks_roi()` |
| Sharpness | `processing/image_processing.py → sharpness()`, `laplacian_edges()` |
| Mask + morphology | `processing/image_processing.py → binary_person_mask()` |
| Background focus | `processing/image_processing.py → suppress_background()` |
| Pose model | `core/pose_detector.py → PoseDetector.__init__()`, `detect()` |
| Main person | `core/pose_detector.py → prominence()` |
| Landmark names, skeleton | `core/pose_detector.py → LANDMARK_NAMES`, `LM`, `SKELETON` |
| Features | `core/feature_extractor.py → FeatureExtractor.extract()` |
| Body side | `core/feature_extractor.py → FeatureExtractor._update_side()` |
| EMA | `core/feature_extractor.py → FeatureExtractor._smooth()` |
| Joint angle | `core/angle_calculator.py → calculate_angle()` |
| Inclination | `core/angle_calculator.py → angle_from_vertical()` |
| Hip offset | `core/angle_calculator.py → signed_distance_to_line()` |
| Recognition | `core/exercise_recognizer.py → ExerciseRecognizer.classify()`, `update()`, `_ramp()` |
| Rep counting | `core/rep_counter.py → RepCounter.update()`, `_zone()` |
| Tracker (counter + analyzer) | `core/pipeline.py → ExerciseTracker.update()`, `phase` |
| Rep scoring | `core/form_analyzer.py → FormAnalyzer.evaluate_rep()`, `median_filter()` |
| Live checks | `core/form_analyzer.py → FormAnalyzer.live()` |
| Score formulas | `core/scoring.py → check_score()`, `weighted_score()`, `grade()`, `status()` |
| Feedback and status | `core/pipeline.py → AnalysisPipeline._compose_feedback()` |
| Warnings | `core/pipeline.py → AnalysisPipeline._collect_warnings()` |
| Frame log | `core/pipeline.py → AnalysisPipeline._log()` |
| Session statistics | `core/session.py → SessionLog.summary()`, `check_breakdown()`, `angle_statistics()`, `issue_counts()`, `reps_df()` |
| Overlay | `processing/visualization.py → draw_overlay()`, `draw_skeleton()`, `draw_angles()` |
| Exercise definitions | `exercises/squat.py → SQUAT`, `bicep_curl.py → BICEP_CURL`, `pushup.py → PUSHUP`, `shoulder_press.py → SHOULDER_PRESS` |
| Check/exercise classes | `exercises/base_exercise.py → FormCheck`, `Exercise`, `TEMPO_CHECK` |
| Thresholds | `config/thresholds.py → EXERCISE_THRESHOLDS`, `PREPROCESSING`, `POSE`, `FEATURES`, `RECOGNITION`, `SCORE_BANDS` |
| Demo clips | `demo/synthetic_athlete.py → SCENARIOS`, `render_scenario()`, `ensure_samples()` |
| DIP experiment | `demo/dip_experiment.py → main()` |
| Export | `utils/export.py → report_csv()`, `report_pdf()` |
| Model download | `utils/helpers.py → ensure_pose_model()` |

## 45.1 Tests (`python -m pytest`, 86 tests)

| File | Tests | What it proves |
|---|---|---|
| `tests/test_angles.py` | 7 | known angle values, scale/translation invariance, symmetry, coincident points raise, `angle_from_vertical` in image coordinates, hip offset positive below the line, distance |
| `tests/test_rep_counter.py` | 10 | clean reps, noise creates no false counts, jitter around one threshold never counts, shallow movement = partial (not counted), starting mid-rep not counted, short gaps hold the state, long absence discards the rep, press starts flexed, too-fast reps rejected, invalid thresholds raise |
| `tests/test_scoring.py` | 10 | check score (both directions, clipping), equal thresholds rejected, weighted score ignores unmeasured checks, grade bands, status colours, good squat scores high, shallow/leaning squat detected, single-frame spike filtered, median filter, live checks use posture rules only |
| `tests/test_exercises.py` | 11 | registry, definitions consistent with config, standing features, right-angle knee, missing landmarks give `None`, no person gives empty features, EMA smoothing, recognizer classifies geometry, idle is unknown and sticky, needs enough samples, tracker state transitions and rep result |
| `tests/test_preprocessing.py` | 10 | resize keeps aspect and never upscales, RGB conversion, dark frame triggers CLAHE, well-lit frame left alone, noise estimate and denoise, dark noisy frame denoised before CLAHE, empty frame raises, sharpness drops with blur, mask morphology removes specks, ROI from landmarks |
| `tests/test_pipeline_integration.py` | 7 (one parametrised over the four exercises) | real MediaPipe on rendered clips: auto detection counts good reps, bad form detected, multiple people reported and the primary analysed, no-person video does not crash, invalid video gives a friendly error, exports contain the session data, unknown manual exercise rejected |
| `tests/test_app.py` | 8 (one parametrised over the 7 pages) | every page renders, top navigation, home buttons and source switcher, `?page=` deep link, demo analysis → report, manual mode kept across pages, empty Results hint, missing camera shows an error |

(7 + 10 + 10 + 11 + 10 + 7 + 8 = 63 test functions; parametrisation expands them to 86 test cases.)

---

# 46. Code snippets, explained line by line

## 46.1 Angle calculation (`core/angle_calculator.py`)

```python
def calculate_angle(point_a, point_b, point_c) -> float:
    bax, bay = point_a[0] - point_b[0], point_a[1] - point_b[1]   # vector BA = A − B
    bcx, bcy = point_c[0] - point_b[0], point_c[1] - point_b[1]   # vector BC = C − B
    norm = math.hypot(bax, bay) * math.hypot(bcx, bcy)            # |BA| · |BC|
    if norm == 0:                                                  # A or C on top of B:
        raise ValueError("Angle undefined: …")                     # no direction, no angle
    cosine = (bax * bcx + bay * bcy) / norm                        # dot product / lengths = cos θ
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))    # clamp, arccos, radians → degrees
```

## 46.2 Pose detection (`core/pose_detector.py → PoseDetector.detect`)

```python
timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)   # VIDEO mode needs increasing times
self._last_ts = timestamp_ms
h, w = rgb.shape[:2]                                        # processed frame size
image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(rgb))      # wrap the RGB array for MediaPipe
result = self._landmarker.detect_for_video(image, timestamp_ms)   # run the model (tracking)
...
for i, landmarks in enumerate(result.pose_landmarks):       # each detected person
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])   # normalised → pixels
    vis = np.array([lm.visibility for lm in landmarks])          # 0..1 per landmark
    people.append((prominence(pts, vis), i, pts, vis))           # size × visibility
_, idx, pts, vis = max(people, key=lambda p: p[0])          # analyse the most prominent person
```

## 46.3 Preprocessing decisions (`processing/preprocessing.py → preprocess_frame`)

```python
bgr, scale = resize_frame(frame_bgr, max_width or PROCESSING["max_width"])   # ≤ 640 px wide
grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)                  # intensity only
region = grey[y1:y2, x1:x2] if roi is usable else grey       # measure on the person
brightness, contrast = grey_statistics(region)                # mean, std
noise_sigma = estimate_noise_sigma(grey)                      # Immerkær σ
if enhance:
    needs_clahe = brightness < 70 or contrast < 30            # too dark or too flat?
    noise_limit = 1.5 if needs_clahe else 4.0                 # stricter if CLAHE will amplify noise
    if noise_sigma > noise_limit:
        bgr = cv2.GaussianBlur(bgr, (5, 5), 0)                # denoise FIRST
    if needs_clahe:
        bgr = apply_clahe(bgr)                                # then boost contrast (L channel)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)                    # model input
```

(Condensed: the real code reads the thresholds from `PREPROCESSING` and records each operation in
`applied`.)

## 46.4 EMA (`core/feature_extractor.py → _smooth`)

```python
def _smooth(self, name, value):
    if value is None:                      # joint not visible:
        self._ema.pop(name, None)          #   forget history, report unknown
        return None
    prev = self._ema.get(name)
    smoothed = value if prev is None else self.alpha * value + (1 - self.alpha) * prev
    self._ema[name] = smoothed             # remember for the next frame
    return smoothed
```

## 46.5 Recognition scores (`core/exercise_recognizer.py → classify`)

```python
horizontal = _ramp(orientation, 35, 65)          # 0 = standing, 1 = lying
upright = 1.0 - horizontal
knee_moves = _ramp(rom("knee_angle"), 15, 40)     # knee range of motion
elbow_moves = _ramp(rom("elbow_angle"), 15, 40)   # elbow range of motion
hands_high = _ramp(median(wrist_above_shoulder), -0.1, 0.2)
scores = {"pushup": horizontal * max(elbow_moves, 0.6),
          "squat": upright * knee_moves,
          "shoulder_press": upright * elbow_moves * hands_high * (1 - knee_moves),
          "bicep_curl": upright * elbow_moves * (1 - hands_high) * (1 - knee_moves)}
label, conf = max(scores.items(), key=lambda kv: kv[1])   # best exercise
if conf < 0.5: label = "unknown"                          # not sure → don't guess
```

## 46.6 Rep state transition (`core/rep_counter.py → RepCounter.update`, the end of a rep)

```python
elif s == RepState.TO_START:                       # coming back up
    if confirmed == "turn":                        # dipped into the turn zone again (2 frames)
        self.state = RepState.TURN                 #   → back to TURN, no new rep
    elif confirmed == "start":                     # reached the start zone (2 frames)
        self.state = RepState.START
        if t - self._move_start >= self.min_rep_seconds:     # long enough to be real
            self.count += 1                                   # REP + 1
            return RepEvent("rep", self._move_start, t, self._extreme)
```

`confirmed` is the current zone only if it has been the same for `confirm_frames = 2` consecutive
frames; this is the debounce.

## 46.7 A form rule (`exercises/squat.py`)

```python
FormCheck("depth", "Depth", "knee_angle", "min",
          "Go deeper: knee angle only reached {value:.0f} deg", weight=1.5)
```

* `"depth"`: looks up `(95, 140)` in `EXERCISE_THRESHOLDS["squat"]["checks"]`.
* `"Depth"`: label on screen.
* `"knee_angle"`: the feature to measure.
* `"min"`: reduce the rep's (median-filtered) knee angles to their minimum = deepest point.
* message: shown if the score < 75, with the measured value inserted.
* `weight=1.5`: counts 1.5× in the rep score.

## 46.8 Scoring (`core/scoring.py` and `core/form_analyzer.py`)

```python
ratio = (value - zero) / (ideal - zero)          # position between zero (0) and ideal (1)
return max(0.0, min(1.0, ratio)) * 100.0         # clip, scale to 0..100

# in FormAnalyzer.evaluate_rep:
score = None if value is None else self._score(check, value)
if score is not None and score < STATUS_GOOD:    # < 75 → it's an issue
    result.issues.append(check.message.format(value=value))
result.score = weighted_score((result.scores[c.key], c.weight) for c in self.exercise.all_checks)
```

## 46.9 Frame processing (`core/pipeline.py → AnalysisPipeline.process`, condensed)

```python
pre = preprocess_frame(frame_bgr, self._roi, self.max_width, self.enhance)   # 1 DIP
pose = self.detector.detect(pre.rgb, int(t * 1000))                           # 2 pose
if pose.detected:
    self._roi = landmarks_roi(pose.points, pose.visibility, pre.bgr.shape)    # 3 ROI (next frame)
    st.sharpness = sharpness(pre.bgr, self._roi)                              #   blur measure
feats = self.extractor.extract(pose)                                          # 4 features
st.recognition = self.recognizer.update(t, feats.values)                      # 5 recognition
active = self.manual_exercise or recognised label (if not unknown)
for key, tracker in self.trackers.items():                                    # 6-8 all trackers
    result = tracker.update(t, feats.values)
    if result and key == active:  store it as a rep or a partial
st.reps, st.phase = self.rep_count(active), tracker.phase
st.live_checks = tracker.analyzer.live(feats.values)                          # live posture
self._compose_feedback(st, exercise); self._collect_warnings(...)             # messages
annotated = draw_overlay(display.copy(), st)                                  # 9 draw
self._log(st, feats, pre)                                                     # 10 log
return annotated, st
```

---

# 47. Glossary

| Term | Meaning (in this project) |
|---|---|
| **Frame** | one still image of a video; the pipeline processes one frame at a time. |
| **Pixel** | one sample of the image; here 3 bytes (B, G, R), each 0-255. |
| **RGB** | colour order red, green, blue; what MediaPipe expects. |
| **BGR** | colour order blue, green, red; how OpenCV stores images. |
| **Greyscale** | one intensity value per pixel; used for statistics, noise and edges. |
| **LAB** | colour space with L = lightness and a, b = colour; CLAHE is applied to L only. |
| **Landmark / keypoint** | a named body point returned by the pose model (33 of them). |
| **Visibility** | the model's 0-1 confidence that a landmark is visible; < 0.5 = missing here. |
| **Pose estimation** | finding body keypoint positions in an image. |
| **Inference** | running a trained model on new input (here: MediaPipe on a frame). |
| **ROI (region of interest)** | the padded box around the person; statistics and sharpness are measured there. |
| **Joint angle** | the angle at a joint between two limb segments, from three landmarks. |
| **Inclination** | the angle of a body segment from the vertical (e.g. torso lean). |
| **Feature** | a number derived from landmarks (angle or normalised distance) that rules use. |
| **Normalisation** | expressing a value relative to a reference: coordinates 0-1, distances ÷ body-segment length, scores 0-100. |
| **Threshold** | a boundary value that triggers a decision (e.g. knee ≤ 125° = bottom). |
| **Hysteresis** | using two separated thresholds so noise near one value can't toggle a state. |
| **Dead band** | the range between the two hysteresis thresholds where nothing changes. |
| **State machine** | a model that is always in one state and moves between states on defined conditions (WAITING, START, TO_TURN, TURN, TO_START). |
| **Debounce** | requiring a condition on several consecutive frames (here 2) before acting. |
| **Partial rep** | a movement that left the start position but never reached the turn position; reported, not counted. |
| **EMA** | exponential moving average, S = α·X + (1 − α)·S_prev; smooths features. |
| **Median filter** | replaces each value with the median of its neighbours; removes spikes. |
| **Gaussian filter** | weighted-average blur with Gaussian weights; removes noise. |
| **CLAHE** | contrast-limited adaptive histogram equalisation; local contrast enhancement. |
| **Histogram** | count of pixels at each intensity level. |
| **Laplacian** | second-derivative operator; strong at edges. Its variance measures sharpness. |
| **Segmentation** | separating the person (foreground) from the background; here by thresholding a probability map. |
| **Binary mask** | an image with only two values (0 and 255) marking foreground. |
| **Morphology (opening/closing)** | shape operations on binary images: opening removes specks, closing fills holes. |
| **Alpha blending** | mixing two images with per-pixel weights. |
| **FPS** | frames per second; processing FPS measures how fast `process()` runs. |
| **Frame skip** | analysing only every n-th frame of a file to go faster. |
| **Check** | one form rule (`FormCheck`) with a feature, a reduction, thresholds, weight and message. |
| **Live check** | a check evaluated on every single frame for real-time colouring. |
| **Weighted mean** | average where some items count more (weights). |
| **Grade band** | the text label for a score range (e.g. 75-89 Acceptable). |
| **Session log** | the in-memory record of every frame and rep (`SessionLog`). |
| **Session state** | Streamlit's per-tab dictionary that survives reruns. |
| **Rerun** | Streamlit executing `app.py` again after an interaction. |
| **Forward kinematics** | computing joint positions from segment lengths and angles (used to render demo clips). |
| **Rule-based** | decisions made by explicit hand-written conditions, not learned from data. |

---

# 48. If you remember only 10 things

1. **The pipeline order:** frame → preprocess → pose (33 landmarks) → features → recognise →
   count → form checks → score → overlay → report. One class runs it:
   `AnalysisPipeline.process()`.
2. **Preprocessing is adaptive:** resize always; measure brightness, contrast and noise on the
   person; Gaussian denoise and then CLAHE **only when needed**; denoise before CLAHE because CLAHE
   amplifies noise (0 % → 90 % detection in the experiment).
3. **MediaPipe BlazePose lite** gives 33 landmarks (x, y normalised → pixels, plus visibility);
   < 0.5 visibility = missing; the most prominent person is analysed.
4. **Joint angle** = arccos(BA·BC / |BA||BC|); angles and ratios are independent of position,
   distance and resolution.
5. **Features are smoothed** with an EMA (α = 0.5); rep minima/maxima use a 5-sample running
   median.
6. **Recognition is rule-based, not ML:** body horizontal → push-up; knee moves → squat; elbow
   moves with hands high → press, hands low → curl; 2-second window, ≥ 0.5 confidence, majority
   vote over 15 frames.
7. **Rep counting is a hysteresis state machine:** start zone → turn zone → start zone, 2-frame
   confirmation, ≥ 0.6 s; squat 160°/125°. Half movements become "partial reps".
8. **Form checks** reduce one feature over the rep (min/max/range) and score it between an ideal
   (100) and a zero (0) value; below 75 → feedback message with the measured number.
9. **Rep score = weighted mean** of measurable check scores (main check 1.5, tempo 0.5, others 1);
   unmeasurable checks are excluded; grades at 90/75/60.
10. **The UI only displays:** it creates the pipeline, feeds frames, shows `FrameStatus`, and the
    Results page reads the `SessionLog`. Demo clips are synthetic but analysed by the same real
    pipeline, which is how the counts are validated (10/10).

---

# 49. The entire project in one diagram

```text
┌────────────────────────────── INPUT ───────────────────────────────┐
│  Webcam (cv2.VideoCapture, wall-clock t)                           │
│  Video file / demo clip (iterate_video, t = frame / fps, skip n)   │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌──────────────────────── IMAGE PROCESSING (DIP) ────────────────────┐
│  resize ≤640 (INTER_AREA) → grey → brightness/contrast on ROI      │
│  → Immerkær noise σ → [Gaussian 5×5 if noisy] → [CLAHE on L if     │
│  dark/flat] → BGR→RGB                  processing/preprocessing.py │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌──────────────────────────── POSE ──────────────────────────────────┐
│  MediaPipe BlazePose lite, VIDEO mode → 33 landmarks (x·w, y·h,    │
│  visibility) → most prominent person          core/pose_detector.py │
│  → ROI (padded box), Laplacian sharpness, [mask + morphology]      │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌─────────────────────────── FEATURES ───────────────────────────────┐
│  side/view → 15 features: joint angles (arccos of dot product),    │
│  inclinations, body-length ratios → EMA α=0.5                      │
│                                          core/feature_extractor.py │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌─────────────────────────── EXERCISE ───────────────────────────────┐
│  2-s window: orientation, knee/elbow ROM, hand height → 4 fuzzy    │
│  scores → ≥0.5 → 15-frame majority   (or manual selection)         │
│                                        core/exercise_recognizer.py │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────── REPETITION ──────────────────────────────┐
│  WAITING → START → TO_TURN → TURN → TO_START → START (+1 if ≥0.6s) │
│  hysteresis (e.g. 160°/125°), 2-frame confirm, partial reps        │
│                                   core/rep_counter.py, pipeline.py │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌───────────────────────────── FORM ─────────────────────────────────┐
│  per rep: median filter → min/max/range per FormCheck              │
│  per frame: live posture checks            core/form_analyzer.py   │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌──────────────────────────── SCORE ─────────────────────────────────┐
│  s = clip((v−zero)/(ideal−zero),0,1)·100 ; rep = Σwᵢsᵢ / Σwᵢ       │
│  grade bands 100/90/75/60 ; status ≥75 good, ≥50 warning           │
│                                                core/scoring.py     │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌─────────────────────────── FEEDBACK ───────────────────────────────┐
│  on video: skeleton in status colour, angle labels, info panel,    │
│  messages, border               processing/visualization.py        │
│  in the app: live read-out panel, session report, charts, CSV/PDF  │
│  ui/, core/session.py, utils/export.py                             │
└────────────────────────────────────────────────────────────────────┘
```
