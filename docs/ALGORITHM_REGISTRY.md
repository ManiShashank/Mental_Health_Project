# ALGORITHM & FEATURE REGISTRY

This registry tracks every algorithmic method, feature extraction pipeline, and statistical routine utilized within the Multimodal Mental Health Pre-Screening System.

**Status Lifecycle:**
- `PLANNED`: Designed and documented; pending implementation in designated milestone.
- `IMPLEMENTED`: Code authored and integrated into pipeline.
- `TESTED`: Unit and integration test suites written and passing.
- `VALIDATED`: Verified against standard benchmark data and statistical invariants.
- `NEEDS REVISION`: Algorithmic anomaly or edge case failure identified; refactoring pending.

---

## Modality 1: Text & Language Analysis

| ID | Algorithm / Feature | Purpose | Input | Output | Library / Model | Parameters | Implementation File | Test File | Status |
|---|---|---|---|---|---|---|---|---|---|
| NLP-01 | Text Preprocessing & Normalization | Clean, tokenize, lemmatize, and filter stop-words | Raw message string | Clean token sequence | `RuleLexicalNLPProvider` / `spacy` | `lower=True`, `regex_tokenizer` | `backend/app/services/nlp_features/providers/rule_lexical_provider.py` | `backend/tests/test_nlp_features.py` | `TESTED` |
| NLP-02 | TF-IDF Feature Extraction | Lexical feature representation across vocabulary | Preprocessed tokens | 20 canonical TF-IDF n-gram projections | `scikit-learn` / Custom Scaler | `canonical_terms=20` | `backend/app/services/nlp_features/providers/rule_lexical_provider.py` | `backend/tests/test_nlp_features.py` | `TESTED` |
| NLP-03 | Contextual Semantic Embeddings | Dense contextual semantic representations | Raw / Tokenized text | 16-d dense semantic summary embedding | `TransformerNLPProvider` / Projection Hash | `dim=16`, `unit_norm=True` | `backend/app/services/nlp_features/providers/transformer_provider.py` | `backend/tests/test_nlp_features.py` | `TESTED` |
| NLP-04 | VADER Sentiment Analysis | Lexicon-based baseline sentiment scoring | Raw message string | Compound, pos, neu, neg scores | `VADER_LEXICON` Rule Engine | `alpha=15.0`, `negation_window=2` | `backend/app/services/nlp_features/providers/rule_lexical_provider.py` | `backend/tests/test_nlp_features.py` | `TESTED` |
| NLP-05 | Contextual Sentiment Dynamics | Deep contextual sentiment polarity classification | Raw message string | Probabilities (Positive, Neutral, Negative) & Polarity | `RuleLexical` / `Transformer` | `return_all_scores=True` | `backend/app/services/nlp_features/providers/rule_lexical_provider.py` | `backend/tests/test_nlp_features.py` | `TESTED` |
| NLP-06 | Affective Emotion Classification | Multi-class affective state probability distribution | Raw message string | Probabilities across 7 emotion classes | `RuleLexical` / `RoBERTa GoEmotions` | `emotions=7`, `laplace_smoothing=0.1` | `backend/app/services/nlp_features/providers/rule_lexical_provider.py` | `backend/tests/test_nlp_features.py` | `TESTED` |
| NLP-07 | LDA Topic Extraction | Unsupervised thematic topic distribution | Preprocessed token corpus | 5 thematic topic probabilities ($1 \times 5$) | `LatentDirichletAllocation` / Keyword Proportions | `n_components=5`, `alpha=0.2` | `backend/app/services/nlp_features/providers/rule_lexical_provider.py` | `backend/tests/test_nlp_features.py` | `TESTED` |
| NLP-08 | Syntactic & Lexical Statistics | Word count, sentence count, TTR diversity, repetition, pronouns | Raw text & POS tags | 10 numeric linguistic metrics | `RuleLexicalNLPProvider` | `pronoun_types=['1st_sing', '1st_plur', '2nd', '3rd']` | `backend/app/services/nlp_features/providers/rule_lexical_provider.py` | `backend/tests/test_nlp_features.py` | `TESTED` |

---

## Modality 2: Voice & Acoustic Analysis

| ID | Algorithm / Feature | Purpose | Input | Output | Library / Model | Parameters | Implementation File | Test File | Status |
|---|---|---|---|---|---|---|---|---|---|
| AUD-01 | Whisper Speech-to-Text | Automatic Speech Recognition converting audio to text | Audio stream/file (WebM/WAV) | Transcribed text string + language metadata | `openai-whisper` (`base` / `tiny`) / Provider Engine | `language=None` (auto-detect) / `en` | `backend/app/services/speech_to_text/` | `backend/tests/test_speech_to_text.py` | `TESTED` |
| AUD-02 | Fundamental Frequency ($F_0$ Pitch) | Measure vocal pitch distribution, mean, std, jitter | Audio signal ($16\text{ kHz}$) | Mean $F_0$, Std $F_0$, Min, Max, Range, Jitter | `librosa.pyin` | `fmin=65`, `fmax=500`, `sr=16000` | `backend/app/services/acoustic_features/extractors/librosa_extractor.py` | `backend/tests/test_acoustic_features.py` | `TESTED` |
| AUD-03 | Energy / RMS & Shimmer | Compute acoustic energy dynamics and loudness variations | Audio signal ($16\text{ kHz}$) | Mean RMS, Std, Min, Max, Range, Shimmer | `librosa.feature.rms` | `frame_length=2048`, `hop_length=512` | `backend/app/services/acoustic_features/extractors/librosa_extractor.py` | `backend/tests/test_acoustic_features.py` | `TESTED` |
| AUD-04 | Speech Rate & Pause Dynamics | Quantify speaking tempo, total pause duration, pause ratio | Audio signal ($16\text{ kHz}$) | Syllables/sec [Est], Pause Count, Pause Duration | `librosa` / `scipy.signal` | `silence_threshold_db=-30`, `min_pause=0.15s` | `backend/app/services/acoustic_features/extractors/librosa_extractor.py` | `backend/tests/test_acoustic_features.py` | `TESTED` |
| AUD-05 | MFCCs (Cepstral Representation) | Spectral envelope and timbre feature extraction | Audio signal ($16\text{ kHz}$) | 13 MFCC means & standard deviations ($26$-d) | `librosa.feature.mfcc` | `n_mfcc=13`, `n_mels=40` | `backend/app/services/acoustic_features/extractors/librosa_extractor.py` | `backend/tests/test_acoustic_features.py` | `TESTED` |
| AUD-06 | Spectral Descriptors | Spectral centroid, bandwidth, rolloff, flatness, ZCR | Audio signal ($16\text{ kHz}$) | Mean & std of spectral descriptors ($10$-d) | `librosa.feature.spectral_*`, `zero_crossing_rate` | `sr=16000`, `roll_percent=0.85` | `backend/app/services/acoustic_features/extractors/librosa_extractor.py` | `backend/tests/test_acoustic_features.py` | `TESTED` |

---

## Modality 3: Communication & Turn Dynamics

| ID | Algorithm / Feature | Purpose | Input | Output | Library / Model | Parameters | Implementation File | Test File | Status |
|---|---|---|---|---|---|---|---|---|---|
| COM-01 | Interaction Turn-Taking Latency | Measure response delay between dyad participants | Timestamped message logs | Mean, min, max response time in seconds | Custom Engine | Turn switch condition: sender transition | `backend/app/analysis/communication/metrics.py` | `backend/tests/test_communication_metrics.py` | `TESTED` |
| COM-02 | Message Cadence & Density | Message count, length stats, duration, and communication gaps | Session message logs | Total msgs, avg chars/words, duration, gap count | Custom Engine | Gap threshold $\tau = 60\text{s}$ | `backend/app/analysis/communication/metrics.py` | `backend/tests/test_communication_metrics.py` | `TESTED` |
| COM-03 | Dyad Participation Ratio | Compute balance of contribution (words, turns, message ratio) | Session message logs | Per-participant message counts & percentage ratios | Custom Engine | Normalized ratio $\sum R_u = 1.0$ | `backend/app/analysis/communication/metrics.py` | `backend/tests/test_communication_metrics.py` | `TESTED` |
| COM-04 | Conversation Initiation Tracker | Identify initiator of interaction episodes | Longitudinal session logs | Participant initiation frequency & ratio | Custom Python Engine | Inactivity window $\ge 30$ min | `backend/app/analysis/communication/initiation_tracker.py` | `backend/tests/test_initiation.py` | `PLANNED` |

---

## Personal Baseline, Temporal Analysis & Multimodal Fusion

| ID | Algorithm / Feature | Purpose | Input | Output | Library / Model | Parameters | Implementation File | Test File | Status |
|---|---|---|---|---|---|---|---|---|---|
| BAS-01 | Longitudinal Personal Baseline | Compute individual sample mean $\mu_i$ and Bessel std $\sigma_i$ per feature | Participant history ($N \ge 3$ observations) | Baseline vector $(\mu, \sigma)$ (140-d) | `PersonalBaselineEngine` | Rolling windows (`all_time`, `rolling_10`, `rolling_30`), $N \ge 3$ | `backend/app/services/personal_baseline/engine.py` | `backend/tests/test_personal_baseline.py` | `TESTED` |
| BAS-02 | Z-Score Deviation Engine | Standardize message features against baseline with two-branch zero-variance regularization | Current feature vector & Baseline $(\mu, \sigma)$ | 140-d raw and clipped $z$-scores, aggregate deviation metrics | `PersonalBaselineEngine` | $\epsilon = 10^{-6}$, clipping $[-5.0, +5.0]$ | `backend/app/services/personal_baseline/engine.py` | `backend/tests/test_personal_baseline.py` | `TESTED` |
| FUS-01 | Early Multimodal Feature Fusion | Concatenate aligned communication (10-d), acoustic (59-d), NLP (66-d), and mask (5-d) vectors with canonical min-max scaling | M3, M6, M7 feature blocks | Unified 140-d raw & normalized feature vectors (`v1.0-140d`) | `MultimodalFusionService` / `MultimodalFeatureScaler` | $D=140$, canonical bounds $[0.0, 1.0]$ | `backend/app/services/multimodal_fusion/` | `backend/tests/test_multimodal_fusion.py` | `TESTED` |
| TEM-01 | Longitudinal Temporal Dynamics | Compute modality-mask-aware SMA-3, SMA-5, EMA-3 ($\alpha=0.5$), EMA-5 ($\alpha=1/3$), OLS slope, and RoC | Longitudinal sequence of 140-d observations | Moving averages, normalized trend slopes, rate of change | `PersonalBaselineEngine` | EMA $\alpha = 2/(W+1)$, normalized OLS threshold $\pm 0.01$ | `backend/app/services/personal_baseline/engine.py` | `backend/tests/test_personal_baseline.py` | `TESTED` |
| PRF-01 | Longitudinal Analytical Profiler | Generate descriptive statistical variation profile, modality completeness, top shifts, and temporal trends | Longitudinal M9 records | Structured `AnalyticalReport` DTO (JSON/MD/HTML) | `AnalyticalReportEngine` | Stable ($<1.0\sigma$), Noticeable ($1.0-2.0\sigma$), Significant ($\ge 2.0\sigma$) | `backend/app/services/analytical_report/engine.py` | `backend/tests/test_analytical_report.py` | `TESTED` |
| EXP-01 | Isolation Forest Anomaly Detection | Experimental unsupervised anomaly scoring | Multimodal feature vectors | Anomaly score $[-1.0, 1.0]$ | `scikit-learn` `IsolationForest` | `contamination=0.1`, `random_state=42` | `backend/app/analysis/experimental/isolation_forest.py` | `backend/tests/test_isolation_forest.py` | `PLANNED` |
