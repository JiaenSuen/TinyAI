# Beyond Classical Sequence Models for Limited-Data Predictive Maintenance

*A leakage-resistant study of recurrent, attention-based, convolutional, spectral, patch-based, and state-space-inspired architectures for long-horizon component failure prediction.*


## Introduction

This project studies multivariate sequence classification for predictive maintenance under limited failure data, severe class imbalance, and long prediction horizons. Using the Microsoft Azure Predictive Maintenance dataset, each model receives the previous **168 hours** of machine history and predicts whether no failure or a specific component failure (`comp1`-`comp4`) will occur within the next **72 hours**.

The objective is not only to compare model accuracy, but to examine which temporal inductive biases remain reliable when failure events are rare, windows are highly correlated, and the observable degradation signal changes with warning lead time. Ten recurrent, attention-based, state-space-inspired, convolutional, patch-based, and spectral architectures are evaluated under one leakage-resistant protocol.

> **Implementation note:** several architectures are compact task-oriented adaptations designed for a controlled PyTorch comparison. Mamba-Lite, RetNet, ModernTCN, TimesNet, TSLANet, PatchTST, and LITE-MV should not be interpreted as exact reproductions of their full official implementations.

---

## Task Definition

- **Input:** `[batch, 168 hours, 56 features]`
- **Output classes:** `none`, `comp1`, `comp2`, `comp3`, `comp4`
- **Prediction horizon:** next 72 hours
- **Primary metric:** macro recall
- **Failure-oriented metrics:** failure-only macro recall, lead-time recall, any-failure detection recall, and event-level component recall

Accuracy is reported but is not treated as the primary model-selection metric because the natural test set contains **4,953 normal windows out of 5,277 samples (93.86%)**.

---

## Models

| Model | Key design and experimental role |
|---|---|
| **LSTM** | A two-layer recurrent baseline using the final hidden state. It provides a compact reference for sequential memory without explicit attention or patching. |
| **Transformer** | A standard encoder with learnable positional embeddings and global self-attention. It tests whether unrestricted pairwise temporal interaction benefits component classification. |
| **Linear Transformer** | Replaces quadratic softmax attention with an `ELU + 1` kernel feature map and associative linear attention. It preserves global aggregation while reducing sequence-length scaling. |
| **RetNet** | Uses parallel multi-scale causal retention, rotary relative positions, RMS normalization, and gated feed-forward layers. It evaluates retention-style temporal decay as an alternative to attention. |
| **Mamba-Lite** | A dependency-free Mamba-inspired temporal mixer using causal depthwise convolutions, short/long dilation paths, multiplicative gating, RMSNorm, and attention pooling. It is designed for native PyTorch and Windows rather than as an official Mamba implementation. |
| **ModernTCN** | Uses patch embedding, large-kernel and small-kernel depthwise temporal branches, inverted bottlenecks, and attention pooling. Its design explicitly targets multi-scale local degradation and long temporal context. |
| **PatchTST** | Encodes each variable as temporal patches with channel-independent Transformer layers, followed by lightweight cross-variable attention. It tests whether patch-level abstraction improves long-window modeling. |
| **TimesNet** | Detects dominant periods through FFT and reorganizes temporal features into lightweight two-dimensional period representations. It targets periodic operating conditions and cross-period variation. |
| **TSLANet** | Combines adaptive Fourier-domain filtering with interactive convolutional mixing. It is intended to suppress noisy frequency components while retaining long-range and local degradation patterns. |
| **LITE-MV** | Applies fixed first-difference, second-difference, and smoothing filters before multi-scale dilated depthwise convolution. It introduces strong signal-processing priors with the smallest parameter budget. |

---

## Dataset and Experimental Strategy

### Leakage-resistant splitting

Machines are divided into **70% training, 15% validation, and 15% test sets before window generation**. Candidate splits are searched for a similar component-failure distribution, while machine IDs remain completely disjoint across splits. This is intentionally stricter than random window splitting: overlapping windows from the same machine cannot appear in both training and evaluation data.

This design measures generalization to unseen machines rather than memorization of machine-specific operating ranges.

### Causal feature construction

The four raw telemetry channels—voltage, rotation, pressure, and vibration—are expanded into causal historical features:

- first-order differences;
- 6-, 24-, and 72-hour rolling means;
- 24-hour rolling standard deviations;
- deviations from the 24-hour local mean;
- 24- and 72-hour error counts;
- 720-hour maintenance counts;
- machine age and model metadata.

All rolling statistics use only the current and previous timestamps. Feature normalization is fitted on training machines only and then frozen for validation and test. With all optional Azure tables available, each timestep contains **56 features**.

The feature design is deliberate: limited independent failure events make it inefficient to require every neural architecture to rediscover basic trend, volatility, and event-frequency descriptors from raw signals alone.

### Event-aware window generation

Training windows use a 12-hour stride, while validation and test use a 24-hour stride. Dense training windows improve temporal coverage, but positive samples are grouped by:

`machine × physical failure event × component × lead-time bin`

The lead-time bins are `0-24 h`, `24-48 h`, and `48-72 h`, with at most two training windows retained per event and bin. This prevents a single failure event from creating many pseudo-independent positive examples.

Windows within 24 hours after a failure are excluded because recovery or maintenance states should not be treated as ordinary normal operation.

### Moderate negative sampling

The sampled training set contains **10,013 windows**:

- `none`: 7,152
- `comp1`: 756
- `comp2`: 1,004
- `comp3`: 474
- `comp4`: 627

Normal windows are reduced to approximately 2.5 times the number of positive windows. Hard negatives—unusual operating states or windows near, but outside, the prediction horizon—are retained but capped at 40% of selected normal samples. Easy normal windows are sampled across machines and operating-condition groups.

This preserves realistic class imbalance while avoiding two opposite errors: overwhelming the learner with repetitive normal operation, or constructing an artificially balanced dataset that does not represent deployment conditions. Validation and test distributions are never resampled.

### Training protocol

All models use AdamW with model-specific learning rates, 30 maximum epochs, two warm-up epochs, cosine learning-rate decay, gradient clipping, and early stopping with patience eight.

The loss uses mild effective-number class weighting, focal gamma 1.0, and label smoothing 0.02. The protocol deliberately avoids stacking aggressive oversampling, large inverse-frequency weights, and high-gamma focal loss.

A `none`-versus-failure threshold is selected only on validation data. The search maximizes validation macro recall while requiring at least 75% normal-class recall. The selected threshold is frozen before the final test. Checkpoint selection uses validation macro recall, with failure-only macro recall as a secondary criterion.

---

## Experimental Results

### Overall test performance

The table is ranked by calibrated test macro recall. `R-c1` to `R-c4` denote component-specific recall.

| Rank | Model | Params | Best Epoch | Accuracy | Macro-R | Failure Macro-R | R-none | R-c1 | R-c2 | R-c3 | R-c4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ModernTCN | 222,342 | 3 | 78.59% | **74.98%** | 73.97% | 79.00% | 70.51% | 63.06% | **87.30%** | 75.00% |
| 2 | LITE-MV | 130,998 | 3 | 75.18% | 74.56% | **74.37%** | 75.33% | 70.51% | 64.86% | 85.71% | 76.39% |
| 3 | TSLANet | 272,841 | 3 | 77.18% | 73.78% | 72.83% | 77.57% | 67.95% | 63.06% | 82.54% | 77.78% |
| 4 | TimesNet | 135,430 | 2 | 76.26% | 72.29% | 71.20% | 76.68% | 66.67% | 62.16% | 80.95% | 75.00% |
| 5 | Transformer | 443,653 | 4 | **80.16%** | 71.65% | 69.32% | **80.98%** | 66.67% | 55.86% | 71.43% | **83.33%** |
| 6 | LSTM | 244,741 | 3 | 77.05% | 70.19% | 68.31% | 77.73% | 66.67% | 56.76% | 76.19% | 73.61% |
| 7 | Linear Transformer | 442,117 | 8 | 78.11% | 69.73% | 67.44% | 78.90% | **76.92%** | 53.15% | 61.90% | 77.78% |
| 8 | Mamba-Lite | 196,038 | 3 | 76.54% | 67.83% | 65.44% | 77.35% | 60.26% | 59.46% | 80.95% | 61.11% |
| 9 | PatchTST | 248,454 | 6 | 79.08% | 66.47% | 63.05% | 80.13% | 46.15% | **67.57%** | 74.60% | 63.89% |
| 10 | RetNet | 568,069 | 3 | 74.87% | 42.15% | 33.26% | 77.73% | 32.05% | 19.82% | 49.21% | 31.94% |

### Lead-time performance

Each cell reports **component-correct recall / any-failure detection recall**. Every lead-time interval contains 108 positive windows, corresponding to 108 physical failure events in the test set.

| Model | 0-24 h C / D | 24-48 h C / D | 48-72 h C / D | Event-R |
|---|---:|---:|---:|---:|
| ModernTCN | 100.00% / 100.00% | 89.81% / 98.15% | 26.85% / 49.07% | 100.00% |
| LITE-MV | 100.00% / 100.00% | 89.81% / 96.30% | 28.70% / 50.93% | 100.00% |
| TSLANet | 98.15% / 100.00% | 87.96% / 95.37% | 27.78% / 54.63% | 99.07% |
| TimesNet | 97.22% / 100.00% | 88.89% / 98.15% | 23.15% / 45.37% | 98.15% |
| Transformer | 96.30% / 100.00% | 79.63% / 91.67% | 26.85% / 50.00% | 97.22% |
| LSTM | 99.07% / 100.00% | 82.41% / 92.59% | 18.52% / 36.11% | 100.00% |
| Linear Transformer | 95.37% / 100.00% | 78.70% / 90.74% | 24.07% / 42.59% | 96.30% |
| Mamba-Lite | 97.22% / 97.22% | 69.44% / 85.19% | 25.93% / 54.63% | 97.22% |
| PatchTST | 89.81% / 99.07% | 82.41% / 93.52% | 16.67% / 46.30% | 93.52% |
| RetNet | 44.44% / 54.63% | 28.70% / 43.52% | 20.37% / 37.96% | 45.37% |

### Validation selection and threshold calibration

| Model | Best Val. Macro-R | Threshold | Raw Macro-R | Calibrated Macro-R | Gain | Test Loss |
|---|---:|---:|---:|---:|---:|---:|
| ModernTCN | 74.69% | 0.370 | 72.53% | 74.98% | +2.45 pp | 0.1395 |
| LITE-MV | 74.72% | 0.335 | 73.39% | 74.56% | +1.17 pp | 0.1399 |
| TSLANet | 74.66% | 0.355 | 70.75% | 73.78% | +3.03 pp | 0.1382 |
| TimesNet | 75.25% | 0.320 | 66.81% | 72.29% | +5.48 pp | 0.1236 |
| Transformer | 73.87% | 0.410 | 70.56% | 71.65% | +1.09 pp | 0.1541 |
| LSTM | 72.29% | 0.370 | 67.05% | 70.19% | +3.14 pp | 0.1379 |
| Linear Transformer | 71.89% | 0.430 | 69.60% | 69.73% | +0.13 pp | NaN* |
| Mamba-Lite | 69.84% | 0.435 | 67.33% | 67.83% | +0.50 pp | 0.1930 |
| PatchTST | 68.95% | 0.280 | 62.06% | 66.47% | +4.41 pp | 0.1476 |
| RetNet | 45.65% | 0.370 | 40.98% | 42.15% | +1.17 pp | 0.2292 |

`*` The Linear Transformer produced a finite classification result but a `NaN` loss value. This run should be repeated with loss computation forced to float32 outside autocast before using its loss value in formal comparison.

---

## Experimental Discussion

### 1. ModernTCN provides the strongest overall balance

ModernTCN achieves the highest test macro recall at **74.98%**, with **73.97% failure-only macro recall** and **100% event-level recall**, while using 222,342 parameters. Its large-kernel and local-kernel branches appear well matched to degradation patterns that combine short transients with multi-day trends.

The result is more informative than raw accuracy alone. The standard Transformer reaches the highest accuracy at **80.16%**, but its macro recall is lower at **71.65%**. Under a test set containing 93.86% normal windows, accuracy mainly reflects normal-operation recognition and can conceal weaker component recall.

### 2. LITE-MV is the strongest efficiency result

LITE-MV reaches **74.56% macro recall** and the best failure-only macro recall at **74.37%** with only **130,998 parameters**. It is within 0.42 percentage points of ModernTCN while using approximately 41% fewer parameters.

This supports a central hypothesis of the project: when failure events are limited, explicit signal priors—differences, curvature, smoothing, and multi-scale convolution—can be more useful than simply increasing model capacity.

### 3. Spectral and convolutional biases outperform the heavier retention model

TSLANet and TimesNet obtain **73.78%** and **72.29% macro recall**, respectively. Their results indicate that frequency-selective and periodic representations are useful for industrial telemetry, especially when combined with local temporal processing.

RetNet performs substantially worse at **42.15% macro recall** despite having the largest parameter count. Its poor event-level recall of **45.37%** suggests that the current parallel-retention adaptation is not well matched to this limited-data classification setting. The result also demonstrates that parameter count and architectural novelty do not guarantee stronger predictive-maintenance performance.

### 4. Lead time, not model family, is the dominant limitation

The highest-performing models reach approximately **98-100% component recall in the final 24 hours** and about **88-90% at 24-48 hours**. However, all models decline sharply in the 48-72-hour interval:

- LITE-MV: 28.70%
- TSLANet: 27.78%
- ModernTCN: 26.85%
- Transformer: 26.85%
- TimesNet: 23.15%

This consistent collapse across recurrent, attention, convolutional, patch, and spectral models indicates a data-level limitation rather than a single architectural failure. Many windows labeled as future component failures may still resemble normal operation two to three days before the event.

The present task should therefore be interpreted as two regimes:

1. **Imminent failure recognition (0-24 h):** highly learnable.
2. **Early failure prediction (48-72 h):** weakly observable and substantially more uncertain.

### 5. Event-level recall is high even when window-level recall is moderate

ModernTCN, LITE-MV, and LSTM detect the correct component in at least one window for all 108 test failure events. TSLANet reaches 99.07%, and TimesNet reaches 98.15%.

This is operationally meaningful: a predictive-maintenance system does not require every overlapping window to be correct if it can produce a sufficiently early and stable alert before each event. Nevertheless, event recall alone is optimistic because it does not measure repeated false alarms. A deployment-oriented evaluation should additionally report alert precision, false alarms per machine-day, persistence requirements, and median warning lead time.

### 6. Component-specific difficulty remains asymmetric

Across the leading models, `comp3` is generally the easiest failure type, reaching 87.30% recall with ModernTCN. `comp2` remains comparatively difficult, with top results around 63-65%. This suggests that some component signatures are more distinguishable in the available telemetry and event history than others.

Further improvement may therefore require component-specific context, hierarchical failure detection, or explicit modeling of the relationship between error types, maintenance records, and future component failures.

### 7. Threshold calibration is useful but not sufficient

Validation-selected thresholding improves macro recall by between **0.13 and 5.48 percentage points**, depending on the model. TimesNet benefits the most, while Linear Transformer changes very little.

Calibration helps correct the normal/failure decision boundary, but it does not solve the 48-72-hour representation problem. The remaining error is primarily caused by weak early evidence and component ambiguity rather than only by an inappropriate classification threshold.

---

## Main Findings

1. **ModernTCN is the strongest overall model**, achieving 74.98% macro recall and 100% event-level recall.
2. **LITE-MV offers the best accuracy-efficiency trade-off**, closely matching ModernTCN with the smallest parameter count.
3. **Convolutional and spectral inductive biases are more reliable than model scale alone** in this limited-failure setting.
4. **The final 24 hours are highly predictable**, but 48-72-hour component prediction remains the central unresolved challenge.
5. **Machine-disjoint splitting and natural test imbalance produce a more realistic but substantially harder evaluation** than random window splitting.
6. **Event-level recall exceeds window-level macro recall**, showing that practical alert design should be evaluated separately from per-window classification.

---

## Running the Experiment

```bash
python dataset.py
python train.py
```

Select one model in `train.py`:

```python
MODEL_NAME = "modern_tcn"
```

Available names:

```text
lstm
transformer
linear_transformer
retnet
mamba
modern_tcn
patch_tst
timesnet
tslanet
lite
```

---

## Future Work

Future experiments will move beyond a single benchmark and test whether the observed ranking is stable across different failure mechanisms, sensor modalities, event frequencies, and domain shifts.

The next architecture-level directions are:

- a hierarchical multi-task model for `failure detection + component identification + lead-time estimation`;
- a hybrid large-kernel temporal and adaptive spectral encoder;
- explicit event-context fusion for telemetry, errors, maintenance, and machine metadata;
- probabilistic time-to-failure or survival objectives instead of a fixed 72-hour classification boundary;
- temporal consistency and alert aggregation for lower operational false-alarm rates;
- repeated-seed and cross-dataset evaluation on additional predictive-maintenance and anomaly-detection benchmarks;
- lightweight deployment studies on edge hardware.

The broader goal is not to identify one universally best sequence model, but to develop experimental architectures whose inductive biases remain effective under limited events, imbalanced labels, long warning horizons, and deployment-level constraints.
