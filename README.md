# The Manga Task [Advanced] – Open-Weight Manga Transcription & Speaker Attribution Pipeline

Overview

This repo performs an end-to-end, fully local, open-weight pipeline that: extracts three-page sequences of english manga, detects/segments dialogue or thought balloons/narration boxes, splits compound multi-lobe speech balloons, filters out all non-story text (SFX, number of page, tags for introduction of character, background signage, documents, interfaces, etc.), orders the story line into a right-to-left, panel aware reading order, and makes any name-entity classification (NARRATION, character1, character2,...) across all three pages of each sequence.

No hosted AI inference APIs are used in any step, including preparing the data, during training, or test prediction. Everything is run locally using open-source solutions and open-weight models, specifically: Tesseract 5.5.3 LSTM OCR, OpenCV, PyTorch, scikit-learn, and SciPy.

Environment Setup & Placement of Dataset (Reproducibility)

# 1. System & Python Dependencies

Require the Tesseract OCR engine (eng.traineddata is included in most Tesseract distributions) and the Python dependencies in requirements.txt:

``bash

# macOS (Homebrew)

brew install tesseract

# Ubuntu / Debian

# sudo apt-get update && sudo apt-get install -y tesseract-ocr

# Required python packages (Python 3.10+)

python3 -m pip install -r requirements.txt

`

Finds tesseract executable automatically with shutil.which("tesseract") (or override it by export TESSERACT_BIN=/path/to/tesseract)

# 2. Expected Dataset Directory Layout

Unpack the dataset files you received in the root directory of your project or inside a dataset/ subdirectory (resolvedatapath() in src/features.py takes care of both patterns automatically):

`text

project_root/

requirements.txt

README.md

score.py

sequences.json # (or dataset/sequences.json)

Samplesubmission.jsonl or dataset/samplesubmission.jsonl

development/ # (or dataset/development/)

labels.jsonl

images/

seq_<id>/

01.png

02.png

03.png

Test/# (or dataset/test/)

images/

seq_<id>/

01.png

02.png

03.png

src/

init.py

features.py

models.py

trainandevaluate.py

predict_test.py

models/

storymlpweights.pt

trained_pipeline.pkl

experiments/

experiment_summary.json

val_predictions.jsonl

valscoresfull.json

oof_predictions.jsonl

oof_scores.json

scores.json

test_predictions.jsonl

submission.jsonl

`

# 3. Running Inference, Training & Scoring

`bash

# Run standalone inference on the 15 test sequences (checkpoint has been saved)

Python3 predicttest.py --sequences.json --sample-submission samplesubmission.jsonl --output test_predictions.jsonl

# Re-compute features and cache OCR candidates across all 285 pages

python3 src/features.py

# Re-train all learned models run validation ablation study + 5-fold CV, save checkpoint

python3 trainandevaluate.py

# # Score out-of-fold development predictions with the official scorer

Python3 score.py --references development/labels.jsonl --predictions experiments/oof_predictions.jsonl --output scores.json

`

Architecture & Data-Driven Adaptations

# 1. Multi-Pass Bubble Proposal & Two-Pass Line-Graph Double-Bubble Splitting (src/features.py)

7 Manga are impenetrable to OCR ("textorderscore = 0.3600") when run on entire pages: screentones, hatching, panels, and multi-column page layouts all interfere with the page-level line segmentation;A more serious issue is that manga use compound (double/triple) speech bubbles–two discrete chucks of dialogue whose lobes share a thin waist. When one feeds a compound bubble into OCR as a single crop, Tesseract reads horizontally across both lobes and jams "horizontal" lines from separated sentences into a single, unspeedlized line.

To solve this, src/features.py implements a three-pass region proposal pipeline:

- Pass 1 (White bubbles enclosed by 230 & 218): Find connected components of white interior (compmask), fill external contours (filled), find interior dark strokes (holes= filled & (~compmask) & (img<212)).
- Two-Pass Line-Graph Lobe Clustering + Unmasked Interior Crops Inside the bounding bubble, glyphs are connected linearly using a tight (7, 2) expansion for horizontal offsets in text lines that do not bridge the 8-12px vertical gap between side-by-side lobes in double-bubbles. We then union-graph adjacent lines stack vertically (for >22 px minimum line-overlap, or relative center offset <0.36 w_{max}) with a second pass that merges single-line horizontal captions and re-attaches small segments of punctuation. Most important here, for each lobe we can save the unmasked grayscale interior of filled (only mask out filled==0 outside the bubbled linerates and lineboxes of other lobes in the same bubble) for strokes or letter connections that extend to the bubble boundary.
- Almost a third of dev lines (514 / 1,753) have ..., and 36 lines stand alone as "..." bubbles whose 4x4 px dots are frequently ignored by even basic OCR. We actively search for collinear triples of small square/round holes (2..10 px) to recover standalone and trailing ellipses.
- Pass 2 (Open-Border White-Surround Blocks): Detects all speech balloons with page-border-opening contours, in addition to white-on-black flashback/whisper balloons, using an inversion of dark-background 255 - img text blocks. Pass 3 (White-on-Black Inverted Bubbles): Detects all white-on-black inverted bubbles using only the inversion of dark-background text blocks.
- Every candidate crop on a page is scaled up by 2x (INTER_CUBIC) and pasted as a single vertical strip with 60px white bands in between, resulting in a single tesseract --psm 6 tsv call on the entire page.

# 2. Learned Component 1: Multimodal Story-Text Region Classifier

Candidate extraction from 80 development sequences (240 pages): ~25 regions/ page, only ~34% true story text, only ~66% non-story candidates (translated SFX: SFX: HOOFBEATS or KLACK, page number at margins, character introduction box DAUGHTER OF PRESIDENT OF A SHIP-BUILDING COMPANY, letter/documents/interface, artwork texture).

Align candidate regions on the training split to development/labels.jsonl with token alignment F1 and edit similarity (aligncandidateto_gold) and train a multimodal ensemble (Story Text Classifier):
- Features: Concatenation of 50 dense geometric, topological, panel-relative, border-sharpness, OCR-confidence and lexical features plus a 24-dim TruncatedSVD projection of character 2-4 gram TF-IDF features.
- Models: 3-layer PyTorch Residual MLP (ResidualMLP) with Layer Norm, GELU, Dropout, class-weighted BCEWith Logits Loss trained via AdamW + CosineAnnealingLR ensembled with a Hist Gradient Boosting Classifier and balanced Random Forest Classifier.
- F1) on held-out validation candidates, improving matchedtokencoverage from 0.6919 (heuristic filter) to 0.7669, and textorderscore from 0.6857 to 0.6936.

# 3. Learned Component 2: Panel-Aware Pairwise Reading-Order Ranker

We slice pages into panels using recursive XY-cut gutter detection (detect_panels) and train a pairwise ranking model (Reading Order Ranker) on the ordered bubble pairs (ci, cj) of the training split:
- Pairwise Features (20 dims): Panel index difference, same-panel indicator, same-bubble-lobe order indicator, normalized $(\Delta cx, \Delta cy)$, horizontal overlap ratio, vertical overlap ratio, top/bottom clearance, reading-flow angle.
- Tournament decoding & honest comparison: Lobe sets with an enclosed parentid are grouped in right-to-left lobeidx order first, then pairwise probabilities 0.8599 pairwise validation accuracy 0.9064 AUROC are decoded through exact bitmask dynamic programming ($O(m^2 2^m)$ for $m \le 13$). On the 16-sequence validation split the learned ranker 0.6898 textorderscore marginally loses out on the panel-aware heuristic sort 0.6936 by 0.0038 due to occasional irregular multi-column panel layouts when XY-cut gutter detection over-splits panels, while achieving 0.7101 textorderscore` on 5-fold cross-validation.

# 4. Learned Components 3 & 4: Narration Classifier & Sequence-Wide Speaker Attribution
- Narration Classifier: Learned with ground-truth positive regions and the following features: rectangular border sharpness, extent = area/w * h, pronoun/tense features, and word/character TFIDF SVD features. Validation AUROC of 0.9205 for binary distinction between NARRATION captions and character speech/thoughts.
- Speaker Attribution Model: Submitted system score.py uses balancedjointf1 which considers all 3 pages, aligns predicted speakers to reference speakers with the Hungarian algorithm and imposes a penalty for extra speakers (extraspeakers in the denominator len(goldspeakers) + extra_speakers).
 - We learn a pairwise co-reference classifier (pairgbdt + pairlogreg) on 28 relational features between turn pairs $(ui, uj)$ for each 3-page sequence: (vertical chain continuation, opposite-side panel turn-taking, $A \to B \to A$ alternating turn parity, 24-dim exterior visual context similarity, font height difference, shared vocative name mentions like "HACHIKEN", "PRESIDENT", "DENJI").
 - A sequence-level Ridge regression model estimates the expected number of speakers $K \in [2, 4]$, and constrained average-linkage hierarchical clustering clusters all non-narration turns across the 3 pages into speaker IDs (character1, character2,...).

External Sources & Our Own Contributions

# External Open-Source Tools & Open-Weight Models Cited

1. Tesseract OCR v5.5.3 (eng.traineddata' LSTM Engine) - Smith, R. (2007), An Overview of the Tesseract OCR Engine, ICDAR - we used this as our local word recognition engine on the segmented composite bubble strips.
2. OpenCV (opencv-python) - Bradski, G. (2000), The OpenCV Library. For connected-component labeling, contour hierarchy, morphological operations and cubic upscaling.
3. PyTorch (torch) - Paszke et al. (2019), PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS. We used this library to construct and train our own custom ResidualMLP neural network (models/storymlpweights.pt).
4. Scikit-Learn (scikit-learn) & SciPy (scipy) - Pedregosa et al. (2011) & Virtanen et al. (2020). Utilising the functions Hist Gradient Boosting Classifier, Random Forest Classifier, Logistic Regression, Ridge, Tfidf Vectorizer, TruncatedSVD, and hierarchical linkage clustering.
5. Related work I: concept level work on understanding Manga Matsui et al. (2017), Sketch-based Manga Retrieval using Manga109 Dataset, Multimedia Tools and Applications; Sachdeva & Zisserman (2024), The Manga Whisperer: Automatically Generating Transcriptions for Comics, CVPR.

# Our Own Contributions

- Multi-Pass Candidate Proposal & Unmasked Interior Crop Extraction (src/features.py) Multi-threshold enclosed white-bubble detector that finds interior dark strokes (holes = filled & ~comp_mask) and extracts unmasked grayscale crops inside filled, removing bubble outlines and nearby artwork clutter with no touching character clipping.
- Two-Pass Line-Graph Double-Bubble Splitter (src/features.py) Custom union-find line-graph clustering algorithm with a (7, 2) line kernel that splits side-by-side text columns in compound/double speech-bubbles before OCR and recombines consecutive lobes with parent_id after OCR.
- Geometric Ellipsis Detector & Manga Contraction Cleaner (src/features.py): Designed a topological 3-dot collinearity detector to detect single-glyph ... Balloons (36 in dev) and trailing ellipses (29.3% of dev lines), & dehyphenation & manga OCR contraction normalization (IMA -> I'M A, TLL -> I'LL, /? ->!?)
- Four Task-Adapted Learned Models (src/models.py, trainandevaluate.py): Visual-specific versions of Story Text Classifier, Reading Order Ranker, Narration Classifier, and Speaker Attribution Model trained on the sequence-level development splits.

Experiments, Comparisons & Ablation Results

A deterministic sequence-preserving 80/20 split (64 train sequences = 192 pages, 16 validation sequences = 48 pages, using every 5th sequence from sequences.json so that no pages from the same sequence appear in both train and validation) and 5-fold cross-validation over all 80 development sequences.

# Validation Split Ablations ( 16Sequences/48 Pages)

| System / Experiment | textorderscore | balancedjointf1 | jointf1 | speakeraccuracyonmatched | matchedtokencoverage |
|:--- |:---: |:---: |:---: |:---: |:---: |
| 1. Same Page OCR Baseline (tesseract --psm 11, single speaker) | 0.3600 | 0.1142 | 0.2629 | 0.5889 | 0.3662 |
| 2. Multi-Pass Bubble Proposal + Heuristic Filter + Heuristic Order | 0.6857 | 0.1804 | 0.4299 | 0.5436 | 0.6919 |
| 3. + Learned Story Text Classifier (PyTorch MLP + GBDT + RF) | 0.6936 | 0.1768 | 0.4383 | 0.5541 | 0.7669 |
| 4. + Learned Reading Order Ranker (Bitmask DP) | 0.6898 | 0.1756 | 0.4352 | 0.5530 | 0.7663 |
| 5. + Narration Classifier + Spatial/Turn K-Means Clustering | 0.6898 | 0.2714 | 0.3776 | 0.4845 | 0.7663 |
| 6. Full Adapted Pipeline (Threshold=0.58, Learned Co-Ref + Avg Linkage) | 0.6898 | 0.2912 | 0.4372 | 0.5650 | 0.7663 |
| 7. Full Adapted Pipeline ('Threshold=0.54', Learned Co-Ref + Avg Linkage) | 0.6874 | 0.3127 | 0.4310 | 0.5588 | 0.7671 |

# 5-Fold Out-of-Fold Cross Validation on All 80 Development Sequences (240 Pages)

| Evaluation Split | textorderscore | balancedjointf1 | jointf1 | speakeraccuracyonmatched | matchedtokencoverage |
|:--- |:---: |:---: |:---: |:---: |:---: |
| 5-Fold Cross Validation, all 80 sequences | 0.7101 | 0.3051 | 0.4588 | 0.5966 | 0.7771 |

Critical Analysis: What Worked vs. What Failed

1. What Worked:
 - Detecting text by interior holes (holes = filled & ~compmask) removed bubble outlines and panel borders prior to OCR (textorderscore up from 0.3600 to 0.6857), unmasked filled interior crops (use compmask, filled instead of glyph-masked crops using crop[sublobeg > 0]) with a (7, 2) line kernel eliminated character clipping on touching characters (WAITED, TEACHERS, COMPLETELY, TROUBLE) while resulting in a distinct double-bubble close side-by-side double-lobe separation (seq6b18300406fa878f, seqc780ac6d1dbd1242).
 - Applying geometric/topological features with character n-gram TF-IDF in our PyTorch MLP + GBDT + RF ensemble (0.9865 auroc) differentiated the story dialogue text from all SFX caption, page number, and character intro box text (increased token coverage by +7.5% -- 0.6919 to 0.7669)
 - DifferentiatingNARRATION(0.9205 AUROC) and fixing cluster numbers $K \in [2, 4]$ raised balancedjointf1 from 0.1756 (single-speaker reference) to 0.2912-0.3127 validation and 0.3051 5-fold CV.

2. What Failed or Showed Clear Limitations:
 - In the absence of explicit bubble-tail keypoint detection or character face re-identification, we use spatial coordinates, turn distance, coarse exterior intensity patches, and vocative mentions. This yields a 0.5573 validation AUROC (0.5966 speaker accuracy on matched tokens in 5-fold CV, 0.3051 balancedjointf1) for multi-character scenes (e.g. 5-8 speaker club scenes in Silver Spoon) or cross-page camera cuts where characters switch left/right panel sides. In these scenarios, one character's speech can span multiple cluster IDs or two characters on the same side of different pages can be co-clustered.
 - Aggressive Turn-Alternation Repulsion (complete linkage with hard adjacent-turn repulsion): we forced adjacent turns (|j-i|==1) on the same page to have low same-speaker probability (psame<=0.18) with complete-linkage, a hard constraint. This degraded 5-fold CV balancedjointf1 (0.3051 -> 0.2875) and speakeraccuracyonmatched (0.5966 -> 0.5343) because manga characters often have 2-4 in a row in adjacent panels or monologues (~54% same-speaker adjacent transitions in development/labels.jsonl).
 - Learned Reading-Order Ranker vs. Panel-Aware Heuristic: On the 16-sequence validation split, Reading Order Ranker (0.6898) failed to beat the much more trivial panel-aware heuristic sort (0.6936) since they both depend on the upstream detect_panels XY-cut segmentation and if any non-rectangular or overlapping panels do not have complete-width white gutters, pairwise panel-index features share the same panel boundary errors.
 - Bubbles present on thick radial speed-lines or tight halftone screentones (see Jujutsu Kaisen panels above) still fall apart when subjected to intensity thresholding. These form the bulk of the ~22% unreclaimed token gap.
 - What We Would Try Next

1. We will train a small keypoint/object detector to locate speech bubble tails (tailtipxy), locate character faces/bodies in each panel, associate each bubble with its pointed-to character crop in the panel, and cluster character crops across the 3 pages with an open-weight manga character embedding network (e.g. Magi / contrastive Res Net/ViT embeddings). This will directly fix the 0.5573 pairwise speaker AUROC bottleneck for camera angle changes between pages.
2. Hookwiring Open-Weight Detection & Manga OCR Backbone (YOLOv8 / RT-DETR + TrOCR): Fine-tuning an open-weight object detector on speech/thought/narration bounding boxes and swapping out Tesseract for a manga-specific vision-encoder-decoder OCR model (e.g., Manga-OCR / TrOCR fine-tuned on synthetic manga fonts) would remove the thresholding failures on translucent, spikey, screentone-filled balloons.
