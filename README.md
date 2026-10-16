# The Manga Task [Advanced] — Open-Weight Manga Transcription & Speaker Attribution Pipeline

## Overview

This repository implements an end-to-end, fully local open-weight pipeline that reads three-page English manga sequences, segments dialogue/thought balloons and narration boxes, splits compound multi-lobe speech bubbles, filters out non-story text (SFX, page numbers, character introduction tags, background signage, documents/interfaces), orders story lines in right-to-left panel-aware reading order, and assigns speaker identities (`NARRATION`, `character1`, `character2`, ...) across all three pages in each sequence.

No hosted AI inference APIs are used anywhere in data preparation, training, or test prediction. All components run locally using open-source tools and open-weight models (**Tesseract 5.5.3 LSTM OCR**, **OpenCV**, **PyTorch**, **scikit-learn**, and **SciPy**).

## Environment Setup & Dataset Placement (Reproducibility)

### 1. System & Python Dependencies
Install the Tesseract OCR engine (`eng.traineddata` is included with standard Tesseract installations) and the Python dependencies listed in `requirements.txt`:

```bash
# macOS (Homebrew)
brew install tesseract

# Ubuntu / Debian
# sudo apt-get update && sudo apt-get install -y tesseract-ocr

# Python packages (Python 3.10+)
python3 -m pip install -r requirements.txt
```

The code locates `tesseract` automatically via `shutil.which("tesseract")` (or you can override it via `export TESSERACT_BIN=/path/to/tesseract`).

### 2. Expected Dataset Directory Layout
Place the supplied dataset files either directly in the project root directory or inside a `dataset/` subdirectory (`resolve_data_path()` in `src/features.py` supports both layouts automatically):

```text
project_root/
├── requirements.txt
├── README.md
├── score.py
├── sequences.json                 # (or dataset/sequences.json)
├── sample_submission.jsonl        # (or dataset/sample_submission.jsonl)
├── development/                   # (or dataset/development/)
│   ├── labels.jsonl
│   └── images/
│       └── seq_<id>/
│           ├── 01.png
│           ├── 02.png
│           └── 03.png
├── test/                          # (or dataset/test/)
│   └── images/
│       └── seq_<id>/
│           ├── 01.png
│           ├── 02.png
│           └── 03.png
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── models.py
├── train_and_evaluate.py
├── predict_test.py
├── models/
│   ├── story_mlp_weights.pt
│   └── trained_pipeline.pkl
├── experiments/
│   ├── experiment_summary.json
│   ├── val_predictions.jsonl
│   ├── val_scores_full.json
│   ├── oof_predictions.jsonl
│   └── oof_scores.json
├── scores.json
├── test_predictions.jsonl
└── submission.jsonl
```

### 3. Running Inference, Training & Scoring

```bash
# Run standalone inference on the 15 test sequences using the saved checkpoint
python3 predict_test.py --sequences sequences.json --sample-submission sample_submission.jsonl --output test_predictions.jsonl

# Extract features and cache OCR candidates from scratch across all 285 pages
python3 src/features.py

# Re-train all learned models, run the validation ablation study + 5-fold CV, and save checkpoints
python3 train_and_evaluate.py

# Score out-of-fold development predictions with the official scorer
python3 score.py --references development/labels.jsonl --predictions experiments/oof_predictions.jsonl --output scores.json
```

## Architecture & Data-Driven Adaptations

### 1. Multi-Pass Bubble Proposal & Two-Pass Line-Graph Double-Bubble Splitting (`src/features.py`)

Running OCR directly on full manga pages fails (`text_order_score = 0.3600`) because screentones, hatching, panel borders, and multi-column layouts corrupt page-level line segmentation. Furthermore, manga frequently uses **compound (double/triple) speech bubbles**—two dialogue lobes connected by a narrow bridge. Passing a compound bubble to OCR as a single crop causes Tesseract to read horizontally across both lobes at once, interleaving words from separate sentences.

To solve this, `src/features.py` implements a three-pass region proposal pipeline:
- **Pass 1 (Enclosed White Bubbles at Dual Thresholds 230 & 218)**: Extracts connected components of white interiors, fills external contours (`filled`), and locates interior dark strokes (`holes = filled & (~comp_mask) & (img < 212)`).
- **Two-Pass Line-Graph Lobe Clustering + Unmasked Interior Crops**: Inside each enclosed bubble, dark glyphs are dilated horizontally with a compact `(7, 2)` kernel to form text line segments without bridging the `8–12px` vertical gap between side-by-side lobes in double bubbles. A graph union-find groups lines that stack vertically ($\Delta y \le 22\text{px}$ with $\ge 32\%$ horizontal overlap or $\le 0.36 w_{\max}$ center offset), followed by a second pass that merges single-line horizontal captions and re-attaches small punctuation fragments. Crucially, for each lobe we extract the **unmasked grayscale interior of `filled`** (masking only `filled == 0` outside the bubble and line boxes belonging to *other* lobes in the same bubble) so that touching letters or anti-aliased strokes near the bubble edge are never clipped.
- **Geometric Ellipsis Recovery**: Nearly 30% of development lines (`514 / 1,753`) contain `...`, and `36` lines are standalone `"..."` bubbles whose `4x4 px` dots are often dropped by standard OCR. We explicitly detect collinear triples of small square/round holes (`2..10 px`) to recover standalone and trailing ellipses.
- **Pass 2 (Open-Border White-Surround Blocks)** & **Pass 3 (White-on-Black Inverted Bubbles)**: Captures speech bubbles whose borders open into page gutters as well as white-on-black flashback/whisper balloons by inverting dark-background glyph clusters (`255 - img`).
- **Batched Composite-Strip OCR**: All candidate crops on a page are upscaled $2\times$ (`INTER_CUBIC`) and stacked into a single vertical strip separated by `60px` white bands, requiring only **one** `tesseract --psm 6 tsv` call per page.

### 2. Learned Component 1: Multimodal Story-Text Region Classifier

From the 80 development sequences (`240` pages), candidate extraction produces `~25` regions per page, of which only `~34%` are true story text and `~66%` are non-story candidates (translated SFX like `SFX: HOOFBEATS` or `KLACK`, page numbers at margins, character introduction boxes like `DAUGHTER OF PRESIDENT OF A SHIP-BUILDING COMPANY`, letters/documents/interfaces, and artwork textures).

We align candidate regions on the training split to `development/labels.jsonl` using token alignment F1 and edit similarity (`align_candidate_to_gold`) and train a multimodal ensemble (`StoryTextClassifier`):
- **Features**: 50 dense geometric, topological, panel-relative, border-sharpness, OCR-confidence, and lexical features concatenated with a 24-dim TruncatedSVD projection of character 2–4 gram TF-IDF features.
- **Models**: A 3-layer **PyTorch Residual MLP (`ResidualMLP`)** with LayerNorm, GELU, Dropout, and class-weighted `BCEWithLogitsLoss` trained via AdamW + CosineAnnealingLR, ensembled with a `HistGradientBoostingClassifier` and balanced `RandomForestClassifier`.
- **Validation Performance**: Achieves **0.9865 AUROC** (`0.9008 F1`) on held-out validation candidates, lifting `matched_token_coverage` from `0.6919` (heuristic filter) to `0.7669` and `text_order_score` from `0.6857` to `0.6936`.

### 3. Learned Component 2: Panel-Aware Pairwise Reading-Order Ranker

We segment pages into panels via recursive XY-cut gutter detection (`detect_panels`) and train a pairwise ranking model (`ReadingOrderRanker`) on ordered bubble pairs `(ci, cj)` from the training split:
- **Pairwise Features (20 dims)**: Panel index difference, same-panel indicator, same-bubble lobe order, normalized $(\Delta cx, \Delta cy)$, horizontal and vertical overlap ratios, top/bottom clearance, and reading-flow angle.
- **Tournament Decoding & Honest Comparison**: Lobes sharing an enclosed `parent_id` are grouped in right-to-left `lobe_idx` order first, and pairwise probabilities (`0.8599` pairwise validation accuracy, `0.9064` AUROC) are decoded via exact bitmask dynamic programming ($O(m^2 2^m)$ for $m \le 13$). On the 16-sequence validation split, the learned ranker (`0.6898` `text_order_score`) slightly trails the panel-aware heuristic sort (`0.6936`) by `0.0038` due to occasional irregular multi-column panel layouts where XY-cut gutter detection over-splits panels, while achieving `0.7101` `text_order_score` across 5-fold cross-validation.

### 4. Learned Components 3 & 4: Narration Classifier & Sequence-Wide Speaker Attribution

- **`NarrationClassifier`**: Trained on ground-truth positive regions using rectangular border sharpness, extent $\frac{\text{area}}{w \cdot h}$, pronoun/tense indicators, and word/character TF-IDF SVD features. Achieves **0.9205 validation AUROC** in separating `NARRATION` captions from character speech/thoughts.
- **`SpeakerAttributionModel`**: Evaluated under `score.py`, `balanced_joint_f1` matches predicted speaker clusters to reference speakers across all 3 pages via the Hungarian algorithm and penalizes unmatched extra speaker clusters (`extra_speakers` in the denominator `len(gold_speakers) + extra_speakers`).
  - We train a pairwise co-reference classifier (`pair_gbdt` + `pair_logreg`) on 28 relational features between turn pairs $(u_i, u_j)$ across each 3-page sequence (vertical chain continuation, opposite-side panel turn-taking, $A \to B \to A$ alternating turn parity, 24-dim exterior visual context similarity, font height difference, and shared vocative name mentions like `"HACHIKEN"`, `"PRESIDENT"`, `"DENJI"`).
  - A sequence-level `Ridge` regression model predicts the expected number of speakers $K \in [2, 4]$, and constrained average-linkage hierarchical clustering partitions all non-narration turns across the 3 pages into speaker IDs (`character1`, `character2`, ...).

## External Sources & Our Own Contributions

### External Open-Source Tools & Open-Weight Models Cited
1. **Tesseract OCR v5.5.3 (`eng.traineddata` LSTM Engine)** — Smith, R. (2007), *An Overview of the Tesseract OCR Engine*, ICDAR. Used as our local word recognition engine on segmented composite bubble strips.
2. **OpenCV (`opencv-python`)** — Bradski, G. (2000), *The OpenCV Library*. Used for connected-component labeling, contour hierarchy analysis, morphological operations, and cubic image upscaling.
3. **PyTorch (`torch`)** — Paszke et al. (2019), *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS. Used to build and train our custom `ResidualMLP` neural network (`models/story_mlp_weights.pt`).
4. **Scikit-Learn (`scikit-learn`) & SciPy (`scipy`)** — Pedregosa et al. (2011) & Virtanen et al. (2020). Used for `HistGradientBoostingClassifier`, `RandomForestClassifier`, `LogisticRegression`, `Ridge`, `TfidfVectorizer`, `TruncatedSVD`, and hierarchical linkage clustering.
5. **Conceptual Prior Work on Manga Understanding** — Matsui et al. (2017), *Sketch-based Manga Retrieval using Manga109 Dataset*, Multimedia Tools and Applications; Sachdeva & Zisserman (2024), *The Manga Whisperer: Automatically Generating Transcriptions for Comics*, CVPR.

### Our Own Contributions
- **Multi-Pass Candidate Proposal & Unmasked Interior Crop Extraction (`src/features.py`)**: Designed a dual-threshold enclosed white-bubble detector that locates interior dark strokes (`holes = filled & ~comp_mask`) and extracts unmasked grayscale crops inside `filled`, eliminating bubble outlines and adjacent artwork clutter without clipping touching characters.
- **Two-Pass Line-Graph Double-Bubble Splitter (`src/features.py`)**: Developed a custom union-find line-graph clustering algorithm using a `(7, 2)` line kernel that separates side-by-side text columns inside compound/double speech bubbles before OCR and reunites consecutive lobes via `parent_id` post-OCR.
- **Geometric Ellipsis Detector & Manga Contraction Cleaner (`src/features.py`)**: Built a topological 3-dot collinearity detector to recover standalone `...` balloons (`36` in dev) and trailing ellipses (`29.3%` of dev lines), plus de-hyphenation and manga OCR contraction normalization (`IMA -> I'M A`, `TLL -> I'LL`, `/? -> !?`).
- **Four Task-Adapted Learned Models (`src/models.py`, `train_and_evaluate.py`)**: Designed, trained, and ablated `StoryTextClassifier`, `ReadingOrderRanker`, `NarrationClassifier`, and `SpeakerAttributionModel` on the sequence-level development splits.

## Experiments, Comparisons & Ablation Results

We use a deterministic sequence-preserving 80/20 split (`64` train sequences = `192` pages; `16` validation sequences = `48` pages, taking every 5th sequence from `sequences.json` so no pages from the same sequence appear in both train and validation), plus **5-fold cross-validation across all 80 development sequences**.

### Validation Split Ablations (16 Sequences / 48 Pages)

| System / Experiment | `text_order_score` | `balanced_joint_f1` | `joint_f1` | `speaker_accuracy_on_matched` | `matched_token_coverage` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Unmodified Full-Page OCR Baseline** (`tesseract --psm 11`, single speaker) | 0.3600 | 0.1142 | 0.2629 | 0.5889 | 0.3662 |
| **2. Multi-Pass Bubble Proposal + Heuristic Filter + Heuristic Order** | 0.6857 | 0.1804 | 0.4299 | 0.5436 | 0.6919 |
| **3. + Learned `StoryTextClassifier` (PyTorch MLP + GBDT + RF)** | **0.6936** | 0.1768 | **0.4383** | 0.5541 | **0.7669** |
| **4. + Learned `ReadingOrderRanker` (Bitmask DP)** | 0.6898 | 0.1756 | 0.4352 | 0.5530 | 0.7663 |
| **5. + `NarrationClassifier` + Spatial/Turn K-Means Clustering** | 0.6898 | 0.2714 | 0.3776 | 0.4845 | 0.7663 |
| **6. Full Adapted Pipeline (`Threshold=0.58`, Learned Co-Ref + Avg Linkage)** | 0.6898 | 0.2912 | 0.4372 | **0.5650** | 0.7663 |
| **7. Full Adapted Pipeline (`Threshold=0.54`, Learned Co-Ref + Avg Linkage)** | 0.6874 | **0.3127** | 0.4310 | 0.5588 | 0.7671 |

### 5-Fold Out-of-Fold Cross-Validation Across All 80 Development Sequences (240 Pages)

| Evaluation Split | `text_order_score` | `balanced_joint_f1` | `joint_f1` | `speaker_accuracy_on_matched` | `matched_token_coverage` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **5-Fold Cross-Validation (All 80 Sequences)** | **0.7101** | **0.3051** | **0.4588** | **0.5966** | **0.7771** |

## Critical Analysis: What Worked vs. What Failed

1. **What Worked**:
   - **Enclosed Interior Hole Detection + Unmasked `filled` Crops**: Detecting text via interior holes (`holes = filled & ~comp_mask`) eliminated bubble outlines and panel borders before OCR (`text_order_score` jumped from `0.3600` to `0.6857`), and switching from glyph-masked crops (`crop[sub_lobe_g > 0]`) to unmasked `filled` interior crops with a `(7, 2)` line kernel eliminated character clipping on touching letters (`WAITED`, `TEACHERS`, `COMPLETELY`, `TROUBLE`) and cleanly separated side-by-side double-bubble lobes (`seq_6b18300406fa878f`, `seq_c780ac6d1dbd1242`).
   - **Multimodal Story-Text Classification**: Combining geometric/topological features with character n-gram TF-IDF in our PyTorch MLP + GBDT + RF ensemble (`0.9865` AUROC) separated story dialogue from SFX captions, page numbers, and character introduction boxes, boosting token coverage by `+7.5%` (`0.6919 -> 0.7669`).
   - **Narration Classification & Speaker Count Regularization**: Separating `NARRATION` (`0.9205` AUROC) and constraining cluster counts $K \in [2, 4]$ lifted `balanced_joint_f1` from `0.1756` (single-speaker baseline) to `0.2912–0.3127` on validation and `0.3051` on 5-fold CV.

2. **What Failed or Showed Clear Limitations**:
   - **Weak Pairwise Speaker Co-Reference Discrimination (`Val AUROC = 0.5573`)**: Without explicit bubble-tail keypoint detection or character face re-identification, our pairwise speaker model relies on spatial coordinates, turn distance, coarse exterior intensity patches, and vocative mentions. This only achieves `0.5573` validation AUROC (`0.5966` speaker accuracy on matched tokens in 5-fold CV, `0.3051` `balanced_joint_f1`). In multi-character scenes (such as 5–8 speaker club scenes in *Silver Spoon*) or cross-page camera cuts where characters swap left/right panel sides, one character's dialogue can be split across multiple cluster IDs or two characters on the same side of different panels can be merged into the same cluster.
   - **Aggressive Turn-Alternation Repulsion (`complete` linkage with hard adjacent-turn repulsion)**: We tested forcing adjacent turns on the same page (`|j - i| == 1`) to have low same-speaker probability (`p_same <= 0.18`) with complete-linkage clustering. That experiment degraded 5-fold CV `balanced_joint_f1` from `0.3051 -> 0.2875` and `speaker_accuracy_on_matched` from `0.5966 -> 0.5343` because manga characters frequently speak **2 to 4 consecutive bubbles** across adjacent panels or monologues (~54% same-speaker adjacent transitions in `development/labels.jsonl`).
   - **Learned Reading-Order Ranker vs. Panel-Aware Heuristic**: On the 16-sequence validation split, `ReadingOrderRanker` (`0.6898`) did not outperform the simpler panel-aware heuristic sort (`0.6936`) because both rely on the upstream `detect_panels` XY-cut segmentation; when non-rectangular or overlapping panels lack full-width white gutters, pairwise panel-index features inherit the same panel boundary errors.
   - **Screentone & Spiky Action Balloons**: Bubbles overlaid on heavy radial speed-lines or dense halftone screentones (e.g., *Jujutsu Kaisen* action panels) still fragment under intensity thresholding, accounting for most of the remaining ~22% unrecovered token gap.

## What We Would Try Next

1. **Bubble-Tail Keypoint Detection + Character Face Re-Identification (Fixing Speaker Attribution)**: Train a lightweight keypoint/object detector to localize speech bubble tails (`tail_tip_xy`) and character faces/bodies within each panel, link each bubble to its pointed character crop within the panel, and cluster character crops across the 3 pages using an open-weight manga character embedding network (e.g., Magi / contrastive ResNet/ViT embeddings). This directly addresses the `0.5573` pairwise speaker AUROC bottleneck when camera angles flip across pages.
2. **Fine-Tuned Open-Weight Detection & Manga OCR Backbone (YOLOv8 / RT-DETR + TrOCR)**: Fine-tuning an open-weight object detector on speech/thought/narration bounding boxes and replacing Tesseract with a manga-adapted Vision-Encoder-Decoder OCR model (such as Manga-OCR / TrOCR fine-tuned on synthetic manga fonts) would eliminate thresholding failures on translucent, spiky, and screentone-filled balloons.
