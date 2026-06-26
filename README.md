# ErgoAssist

This repository contains the analysis and real-time implementation code for **ErgoAssist**, a cognition-aware wearable ergonomic assistant. The system combines IMU-based posture sensing with EEG-based cognitive-state estimation to study posture, cognitive load, perceived discomfort, and real-time ergonomic alert scheduling.

The repository includes code for:

* EEG cognitive-load analysis using Frenz Brainband signals
* IMU posture classification using head-tracker data
* Multimodal discomfort classification using EEG, IMU, and survey responses
* Real-time posture-only and cognition-aware alert policies

---

## Repository Files

```text
ErgoAssist/
│
├── EEG_CognitiveLoad.ipynb
├── IMU_Posture.ipynb
├── Multimodal Discomfort Code.ipynb
├── ErgoAssist_Alert.py
├── P-Only Alert.py
└── README.md
```

---

# 1. EEG Cognitive-Load Analysis

File:

```text
EEG_CognitiveLoad.ipynb
```

This notebook contains the EEG-based cognitive-state analysis pipeline using signals collected from the **Frenz Brainband**.

The module includes:

* Frenz band signal-strength analysis
* EEG and EOG preprocessing
* EEG artifact inspection
* EEG feature extraction
* Binary cognitive-state classification
* Multiclass cognitive-state classification
* Grid search
* EEG window ablation study
* EEG feature ablation study
* EEG feature importance analysis
* EEG biomarker and behavior analysis

## 1.1 Frenz Band Signal Strength

This part analyzes signal quality from the Frenz Brainband channels. It is used to inspect whether the EEG channels provide usable signal strength before cognitive-state classification.

Typical outputs include:

* Channel-level signal quality
* Signal-strength summaries
* Visualization of EEG channel reliability
* Identification of noisy or artifact-prone segments

## 1.2 EEG and EOG Preprocessing

The EEG preprocessing pipeline handles raw EEG signals and EOG-related artifacts.

Main steps include:

* Raw EEG loading
* Channel selection
* Boundary trimming
* Band-pass filtering
* EOG and blink-related inspection
* Motion-artifact handling
* Windowing of EEG signals
* EEG feature extraction

The EEG pipeline focuses on cognitively relevant spectral bands, especially theta, alpha, and beta rhythms.

## 1.3 Cognitive-State Classification

The EEG notebook supports both binary and multiclass classification.

### Binary Classification

```text
Rest vs. Active
```

where active includes cognitively demanding tasks such as Stroop and numerical calculation.

### Multiclass Classification

```text
Rest vs. Stroop vs. Numerical Calculation
```

Models include:

* Random Forest
* Support Vector Machine
* Linear SVM
* Gradient Boosting

Evaluation settings include:

* 80/20 train-test split
* Leave-One-Subject-Out cross-validation
* Accuracy
* Macro precision
* Macro recall
* Macro F1-score

## 1.4 Grid Search

The notebook includes hyperparameter search for EEG-based cognitive-state classification.

Search spaces include:

* Random Forest: number of estimators, maximum depth, minimum leaf size
* SVM: regularization parameter and kernel scale
* Linear SVM: regularization strength
* Gradient Boosting: number of estimators, learning rate, and tree depth

## 1.5 EEG Window and Feature Ablation

The notebook evaluates how classification performance changes with different EEG design choices, including:

* EEG window length
* Window overlap
* Channel subsets
* Band-power features
* Relative power features
* Band-ratio features
* Coherence features
* Engagement-index features

## 1.6 EEG Feature Importance

Feature importance is computed to interpret which EEG features contribute most strongly to cognitive-state classification.

Supported analyses include:

* Random Forest feature importance
* Gradient Boosting feature importance
* Linear SVM coefficient-based importance
* Feature ranking across EEG biomarkers

## 1.7 EEG Biomarkers and Behavior

This analysis studies the relationship between EEG-derived biomarkers and task behavior.

Representative biomarkers include:

* Theta power
* Alpha power
* Beta power
* Theta/beta ratio
* Theta/alpha ratio
* Alpha/beta ratio
* Engagement index
* Channel coherence

---

# 2. IMU Posture Analysis

File:

```text
IMU_Posture.ipynb
```

This notebook contains the IMU-based posture classification pipeline using head-tracker data.

The module includes:

* IMU baseline correction
* Posture feature extraction
* Binary posture classification
* Multiclass posture classification
* Grid search
* Feature ablation study
* Feature importance analysis

## 2.1 IMU Baseline Correction

The IMU pipeline performs participant-specific baseline correction to reduce differences caused by natural head posture and sensor placement.

The neutral baseline is computed from the participant’s neutral posture condition:

```text
relative_pitch = raw_pitch - baseline_pitch
```

This produces a baseline-normalized posture signal.

## 2.2 Posture Classification

The notebook supports both binary and multiclass posture classification.

### Binary Classification

```text
Neutral vs. Slouched
```

where slouched combines mild and deep forward-flexion postures.

### Multiclass Classification

```text
D1 vs. D2 vs. D3
```

where:

* `D1`: neutral posture
* `D2`: mild forward flexion
* `D3`: deep forward flexion

Models include:

* Random Forest
* Support Vector Machine
* Linear SVM
* Gradient Boosting

Evaluation settings include:

* 80/20 train-test split
* Leave-One-Subject-Out cross-validation
* Accuracy
* Macro precision
* Macro recall
* Macro F1-score

## 2.3 IMU Feature Extraction

Representative IMU features include:

* Pitch mean
* Pitch median
* Pitch standard deviation
* Pitch RMS
* Pitch range
* Interquartile range
* Skewness
* Kurtosis
* Pitch velocity
* Pitch acceleration
* Hjorth activity
* Hjorth mobility
* Hjorth complexity
* Dominant frequency
* Spectral entropy
* Low-frequency and high-frequency power ratios

## 2.4 IMU Grid Search

The notebook includes grid search for posture classification models.

The grid search evaluates:

* Random Forest configurations
* SVM configurations
* Linear SVM configurations
* Gradient Boosting configurations

## 2.5 IMU Feature Ablation

The ablation study evaluates the contribution of different IMU feature groups.

Possible groups include:

* Time-domain features
* Derivative-based features
* Frequency-domain features
* Hjorth features
* Pitch-only features
* Full IMU feature set

## 2.6 IMU Feature Importance

Feature importance analysis is used to identify the most informative posture features.

Supported methods include:

* Random Forest feature importance
* Gradient Boosting feature importance
* Linear SVM coefficient-based importance

---

# 3. Multimodal Discomfort Analysis

File:

```text
Multimodal Discomfort Code.ipynb
```

This notebook contains perceived discomfort analysis using survey responses, IMU features, EEG features, and multimodal fusion.

The module includes:

* Survey analysis
* CMDQ analysis
* NASA-TLX analysis
* Unimodal discomfort classification
* Multimodal discomfort classification
* Feature importance analysis
* Latency analysis

## 3.1 Survey Analysis

The survey analysis summarizes participant responses related to:

* Laptop usage
* Phone usage
* Neck pain
* Upper-back discomfort
* Eye strain
* Posture habits
* Perceived usefulness of ergonomic feedback

## 3.2 CMDQ Analysis

The Cornell Musculoskeletal Discomfort Questionnaire is used to quantify physical discomfort.

The analysis includes:

* Neck discomfort
* Shoulder discomfort
* Upper-back discomfort
* Posture-related discomfort trends
* Discomfort across posture conditions

## 3.3 NASA-TLX Analysis

NASA-TLX is used to quantify perceived workload.

The analysis includes:

* Mental demand
* Physical demand
* Temporal demand
* Effort
* Frustration
* Perceived performance

## 3.4 Unimodal Discomfort Classification

Unimodal discomfort classification predicts perceived discomfort using one modality at a time.

Supported settings include:

```text
IMU only
EEG only
```

Models include:

* Random Forest
* Support Vector Machine
* Linear SVM
* Gradient Boosting

Evaluation settings include:

* 80/20 train-test split
* Leave-One-Subject-Out validation
* Accuracy
* Precision
* Recall
* F1-score

## 3.5 Multimodal Discomfort Classification

Multimodal discomfort classification combines EEG and IMU features.

The pipeline includes:

* EEG feature aggregation
* IMU feature aggregation
* Feature concatenation
* Missing-value imputation
* Feature filtering
* Feature selection
* Model training
* Model evaluation

Models include:

* Random Forest
* Support Vector Machine
* Linear SVM
* Gradient Boosting

## 3.6 Feature Importance

Feature importance analysis compares the predictive contribution of:

* EEG-only features
* IMU-only features
* Multimodal features

This helps identify whether discomfort is better explained by posture features, cognitive features, or their combination.

## 3.7 Latency Analysis

The latency analysis evaluates runtime feasibility for near-real-time use.

It includes:

* Feature extraction latency
* Model inference latency
* Alert scheduling latency
* End-to-end runtime overhead

---

# 4. Real-Time Alert Policies

The repository contains two real-time alert implementations:

```text
P-Only Alert.py
ErgoAssist_Alert.py
```

---

## 4.1 Posture-Only Alert

File:

```text
P-Only Alert.py
```

This script implements the posture-only alert policy.

The logic is:

1. Collect a neutral posture baseline.
2. Compute pitch deviation from the personalized baseline.
3. Detect bad posture when the pitch deviation exceeds the posture threshold.
4. Require the bad posture to persist for a fixed duration.
5. Trigger an alert if the posture remains bad and the system is outside the cooldown interval.
6. Reset the bad-posture counter after alerting.

This corresponds to the posture-only alert scheduling policy.

---

## 4.2 Cognition-Aware Alert

File:

```text
ErgoAssist_Alert.py
```

This script implements the cognition-aware alert policy using IMU and EEG.

The policy combines:

* Pretrained IMU posture classifier
* Pretrained EEG cognitive-state classifier
* Personalized baseline normalization
* Sustained bad-posture tracking
* Continuous bad-posture override
* EEG-based interruptibility estimation
* Alert interval control

The real-time logic is:

1. Read the IMU stream.
2. Collect baseline IMU samples.
3. Baseline-normalize the IMU window.
4. Predict posture using the pretrained IMU classifier.
5. Track bad-posture count and duration.
6. Periodically infer cognitive state from EEG.
7. Trigger alert if:

   * bad posture persists beyond the continuous override window, or
   * bad posture is sustained and the EEG state indicates the user is interruptible.
8. Reset bad-posture count and duration after alerting.

This corresponds to the cognition-aware P+C alert scheduling policy.

---

# 5. Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Representative dependencies include:

```text
numpy
pandas
scipy
scikit-learn
imbalanced-learn
matplotlib
joblib
mido
frenztoolkit
```

The real-time alert scripts are designed for macOS and may use:

```text
osascript
afplay
open
```

---

# 6. Data and Model Files

Raw participant data, logs, and trained models should not be committed to GitHub unless explicitly permitted.

Recommended local structure:

```text
data/
├── eeg/
├── imu/
├── survey/
├── processed/
└── features/

models/
├── eeg_models/
└── imu_models/

logs/
```

Recommended `.gitignore` entries:

```text
data/
logs/
models/
*.pkl
*.joblib
*.csv
*.edf
*.mat
*.npy
*.npz
.DS_Store
```

---

# 7. Outputs

Generated outputs may include:

```text
results/
├── eeg_cognitive/
├── imu_posture/
├── multimodal/
└── real_time_alerts/
```

Typical outputs include:

* Classification metrics
* Confusion matrices
* Feature importance tables
* Ablation-study results
* Grid-search results
* Survey summaries
* Latency measurements
* Real-time alert logs
* Generated figures

