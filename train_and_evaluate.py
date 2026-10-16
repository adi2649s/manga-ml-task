#!/usr/bin/env python3
"""
Training, validation, ablation study, and 5-fold cross-validation script
for The Manga Task [Advanced].
"""
from __future__ import annotations

import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from score import evaluate, load_jsonl
from src.features import TESSERACT_BIN, build_or_load_cache
from src.models import (
    NarrationClassifier,
    ReadingOrderRanker,
    SpeakerAttributionModel,
    StoryTextClassifier,
    align_candidate_to_gold,
    pairwise_order_features,
    pairwise_speaker_features,
)


def build_supervision_for_sequences(
    seq_ids: List[str],
    seq_map: Dict[str, Dict[str, Any]],
    labels: Dict[str, List[List[Dict[str, str]]]],
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    all_cands: List[Dict[str, Any]] = []
    y_story_list: List[int] = []

    order_pairs_X: List[np.ndarray] = []
    order_pairs_y: List[int] = []

    pos_cands: List[Dict[str, Any]] = []
    y_narr_list: List[int] = []

    spk_pairs_X: List[np.ndarray] = []
    spk_pairs_y: List[int] = []
    seq_cnt_X: List[np.ndarray] = []
    seq_cnt_y: List[float] = []

    for sid in seq_ids:
        s_info = seq_map[sid]
        gold_pages = labels[sid]
        seq_dialogue_items: List[Tuple[int, int, Dict[str, Any], str]] = []

        for p_idx, img_path in enumerate(s_info["images"]):
            page_data = cache[img_path]
            W, H = page_data["width"], page_data["height"]
            cands = page_data["candidates"]
            gold_lines = gold_pages[p_idx]

            matched_on_page: List[Tuple[Dict[str, Any], int, str]] = []
            for c in cands:
                sc, g_idx = align_candidate_to_gold(c["text"], gold_lines)
                is_pos = 1 if (sc >= 0.46 and g_idx >= 0) else 0
                all_cands.append(c)
                y_story_list.append(is_pos)
                if is_pos:
                    g_spk = gold_lines[g_idx]["speaker"]
                    matched_on_page.append((c, g_idx, g_spk))
                    pos_cands.append(c)
                    y_narr_list.append(1 if g_spk == "NARRATION" else 0)

            n_m = len(matched_on_page)
            for a in range(n_m):
                ca, ga, _ = matched_on_page[a]
                for b in range(a + 1, n_m):
                    cb, gb, _ = matched_on_page[b]
                    if ga == gb:
                        if ca.get("parent_id") == cb.get("parent_id") and ca.get("lobe_idx", 0) != cb.get("lobe_idx", 0):
                            first_a = 1 if ca.get("lobe_idx", 0) < cb.get("lobe_idx", 0) else 0
                        else:
                            first_a = 1 if ca["bbox"][0] > cb["bbox"][0] else 0
                    else:
                        first_a = 1 if ga < gb else 0
                    order_pairs_X.append(pairwise_order_features(ca, cb, W, H))
                    order_pairs_y.append(first_a)
                    order_pairs_X.append(pairwise_order_features(cb, ca, W, H))
                    order_pairs_y.append(1 - first_a)

            matched_on_page.sort(key=lambda item: (item[1], -item[0]["bbox"][0]))
            merged_page_turns: List[Tuple[Dict[str, Any], str]] = []
            for c, g_idx, g_spk in matched_on_page:
                if g_spk == "NARRATION":
                    continue
                if (
                    merged_page_turns
                    and c.get("source") == "enclosed"
                    and merged_page_turns[-1][0].get("source") == "enclosed"
                    and c.get("parent_id") == merged_page_turns[-1][0].get("parent_id")
                ):
                    continue
                merged_page_turns.append((c, g_spk))

            for wp_idx, (c, g_spk) in enumerate(merged_page_turns):
                seq_dialogue_items.append((p_idx, wp_idx, c, g_spk))

        n_dial = len(seq_dialogue_items)
        seq_same_flags = []
        for i in range(n_dial):
            pi, wpi, ui, spk_i = seq_dialogue_items[i]
            for j in range(i + 1, n_dial):
                pj, wpj, uj, spk_j = seq_dialogue_items[j]
                if spk_i in (None, "UNKNOWN") or spk_j in (None, "UNKNOWN"):
                    continue
                same_spk = 1 if spk_i == spk_j else 0
                spk_pairs_X.append(pairwise_speaker_features(ui, uj, i, j, pi, pj, wpi, wpj))
                spk_pairs_y.append(same_spk)
                seq_same_flags.append(same_spk)

        gold_distinct_speakers = {
            g["speaker"]
            for p in gold_pages
            for g in p
            if g["speaker"] not in (None, "UNKNOWN", "NARRATION")
        }
        if n_dial >= 2 and gold_distinct_speakers:
            dial_no_spk = [(p, wp, c) for p, wp, c, _ in seq_dialogue_items]
            mean_same = float(np.mean(seq_same_flags)) if seq_same_flags else 0.35
            seq_cnt_X.append(SpeakerAttributionModel.sequence_count_features(dial_no_spk, mean_same))
            seq_cnt_y.append(float(len(gold_distinct_speakers)))

    return {
        "all_cands": all_cands,
        "y_story": np.array(y_story_list, dtype=np.int32),
        "X_order": np.vstack(order_pairs_X) if order_pairs_X else np.zeros((0, 20), dtype=np.float32),
        "y_order": np.array(order_pairs_y, dtype=np.int32),
        "pos_cands": pos_cands,
        "y_narr": np.array(y_narr_list, dtype=np.int32),
        "X_spk": np.vstack(spk_pairs_X) if spk_pairs_X else np.zeros((0, 28), dtype=np.float32),
        "y_spk": np.array(spk_pairs_y, dtype=np.int32),
        "X_cnt": np.vstack(seq_cnt_X) if seq_cnt_X else np.zeros((0, 8), dtype=np.float32),
        "y_cnt": np.array(seq_cnt_y, dtype=np.float32),
    }


def predict_sequence_with_pipeline(
    sid: str,
    s_info: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    story_clf: Optional[StoryTextClassifier],
    order_ranker: Optional[ReadingOrderRanker],
    narr_clf: Optional[NarrationClassifier],
    spk_model: SpeakerAttributionModel,
    filter_mode: str = "learned",
    order_mode: str = "learned",
    speaker_mode: str = "learned",
) -> List[List[Dict[str, str]]]:
    pages_ordered_cands: List[List[Dict[str, Any]]] = []
    for img_path in s_info["images"]:
        page_data = cache[img_path]
        W, H = page_data["width"], page_data["height"]
        cands = page_data["candidates"]
        if not cands:
            pages_ordered_cands.append([])
            continue

        if filter_mode == "heuristic":
            kept = []
            for c in cands:
                f = c["features"]
                if c.get("source") == "enclosed" and c.get("conf", 0) >= 68.0 and len(c.get("text", "")) >= 2:
                    if f[44] < 0.5 and f[45] < 0.5:
                        kept.append(c)
        else:
            assert story_clf is not None
            probs = story_clf.predict_proba(cands)
            if float(np.max(probs)) < story_clf.empty_page_max_thresh and float(np.sum(probs >= story_clf.threshold)) <= 1:
                kept = []
            else:
                kept = [c for c, p in zip(cands, probs) if p >= story_clf.threshold]

        if not kept:
            pages_ordered_cands.append([])
            continue

        if order_mode == "heuristic" or order_ranker is None:
            ordered = sorted(
                kept,
                key=lambda c: (
                    c.get("panel_idx", 0),
                    (c["bbox"][1] + c["bbox"][3] / 2.0) / H - 0.35 * ((c["bbox"][0] + c["bbox"][2] / 2.0) / W),
                ),
            )
        else:
            ordered = order_ranker.order_candidates(kept, W, H)

        pages_ordered_cands.append(ordered)

    return spk_model.assign_speakers(
        pages_ordered_cands,
        narration_classifier=narr_clf if speaker_mode in ("learned", "spatial_kmeans") else None,
        mode=speaker_mode,
    )


def run_unmodified_fullpage_baseline(
    seq_ids: List[str],
    seq_map: Dict[str, Dict[str, Any]],
    cache_file: Path = Path("cache/unmodified_baseline.pkl"),
) -> Dict[str, List[List[Dict[str, str]]]]:
    if cache_file.exists():
        all_cached = pickle.load(open(cache_file, "rb"))
        if all(sid in all_cached for sid in seq_ids):
            return {sid: all_cached[sid] for sid in seq_ids}
    else:
        all_cached = {}

    for sid in seq_ids:
        if sid in all_cached:
            continue
        pages_out = []
        for img_path in seq_map[sid]["images"]:
            res = subprocess.run(
                [TESSERACT_BIN, img_path, "stdout", "--psm", "11", "tsv"],
                capture_output=True,
                text=True,
                errors="replace",
            )
            blocks: Dict[Tuple[int, int], List[Tuple[int, int, str, float]]] = {}
            for line in res.stdout.splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 12 and parts[0] == "5":
                    w = parts[11].strip()
                    conf = float(parts[10])
                    if w and conf >= 65.0:
                        b_key = (int(parts[2]), int(parts[3]))
                        blocks.setdefault(b_key, []).append((int(parts[6]), int(parts[7]), w, conf))
            b_items = []
            for words_info in blocks.values():
                txt = " ".join(x[2] for x in words_info).strip()
                if len(txt) >= 2 and any(c.isalpha() for c in txt):
                    mx = float(np.mean([x[0] for x in words_info]))
                    my = float(np.mean([x[1] for x in words_info]))
                    b_items.append((my // 100, -mx, txt))
            b_items.sort()
            pages_out.append([{"speaker": "character1", "text": item[2]} for item in b_items])
        all_cached[sid] = pages_out

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(all_cached, f)
    return {sid: all_cached[sid] for sid in seq_ids}


def tune_story_threshold(
    story_clf: StoryTextClassifier,
    order_ranker: ReadingOrderRanker,
    narr_clf: NarrationClassifier,
    spk_model: SpeakerAttributionModel,
    tune_seq_ids: List[str],
    seq_map: Dict[str, Dict[str, Any]],
    labels: Dict[str, List[List[Dict[str, str]]]],
    cache: Dict[str, Dict[str, Any]],
) -> float:
    best_th = 0.54
    best_score = -1.0
    gold_sub = {sid: labels[sid] for sid in tune_seq_ids}
    for th in [0.46, 0.50, 0.54, 0.58]:
        story_clf.threshold = th
        story_clf.empty_page_max_thresh = th + 0.06
        preds = {
            sid: predict_sequence_with_pipeline(
                sid, seq_map[sid], cache, story_clf, order_ranker, narr_clf, spk_model
            )
            for sid in tune_seq_ids
        }
        rep = evaluate(gold_sub, preds)
        comb = 0.50 * rep["macro"]["text_order_score"] + 0.50 * rep["macro"]["balanced_joint_f1"]
        if comb > best_score:
            best_score = comb
            best_th = th
    story_clf.threshold = best_th
    story_clf.empty_page_max_thresh = best_th + 0.06
    return best_th


def save_predictions_jsonl(preds: Dict[str, List[List[Dict[str, str]]]], seq_order: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for sid in seq_order:
        row = {"sequence_id": sid, "pages": preds[sid]}
        lines.append(json.dumps(row, ensure_ascii=False))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t_start = time.time()
    cache = build_or_load_cache()
    seqs = json.loads(Path("sequences.json").read_text(encoding="utf-8"))
    seq_map = {s["sequence_id"]: s for s in seqs}
    dev_seqs = [s for s in seqs if s["split"] == "development"]
    dev_ids = [s["sequence_id"] for s in dev_seqs]
    labels = load_jsonl("development/labels.jsonl", reference=True)
    raw_labels = {
        row["sequence_id"]: row["pages"]
        for row in [json.loads(line) for line in Path("development/labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    }

    val_ids = [dev_ids[i] for i in range(len(dev_ids)) if i % 5 == 4]
    train_ids = [sid for sid in dev_ids if sid not in set(val_ids)]
    print(f"Sequence-level split: {len(train_ids)} train sequences ({len(train_ids)*3} pages), {len(val_ids)} val sequences ({len(val_ids)*3} pages)")

    sup_train = build_supervision_for_sequences(train_ids, seq_map, raw_labels, cache)
    sup_val = build_supervision_for_sequences(val_ids, seq_map, raw_labels, cache)

    print("\n--- Training Learned Models on 64 Training Sequences ---")
    story_clf = StoryTextClassifier(seed=42)
    story_clf.fit(sup_train["all_cands"], sup_train["y_story"])
    val_story_probs = story_clf.predict_proba(sup_val["all_cands"])
    val_story_auroc = float(roc_auc_score(sup_val["y_story"], val_story_probs))
    val_story_f1 = float(f1_score(sup_val["y_story"], (val_story_probs >= 0.54).astype(int)))
    print(f"[Model 1] StoryTextClassifier Val AUROC = {val_story_auroc:.4f}, F1 = {val_story_f1:.4f}")

    order_ranker = ReadingOrderRanker(seed=42)
    order_ranker.fit(sup_train["X_order"], sup_train["y_order"])
    val_order_X = order_ranker.scaler.transform(sup_val["X_order"])
    val_order_probs = 0.65 * order_ranker.gbdt.predict_proba(val_order_X)[:, 1] + 0.35 * order_ranker.logreg.predict_proba(val_order_X)[:, 1]
    val_order_acc = float(np.mean((val_order_probs >= 0.5) == sup_val["y_order"]))
    val_order_auroc = float(roc_auc_score(sup_val["y_order"], val_order_probs))
    print(f"[Model 2] ReadingOrderRanker Pairwise Val Acc = {val_order_acc:.4f}, AUROC = {val_order_auroc:.4f}")

    narr_clf = NarrationClassifier(seed=42)
    narr_clf.fit(sup_train["pos_cands"], sup_train["y_narr"])
    val_narr_probs = narr_clf.predict_proba(sup_val["pos_cands"])
    val_narr_auroc = float(roc_auc_score(sup_val["y_narr"], val_narr_probs)) if len(np.unique(sup_val["y_narr"])) > 1 else 0.0
    print(f"[Model 3] NarrationClassifier Val AUROC = {val_narr_auroc:.4f}")

    spk_model = SpeakerAttributionModel(seed=42)
    spk_model.fit(sup_train["X_spk"], sup_train["y_spk"], sup_train["X_cnt"], sup_train["y_cnt"])
    val_spk_X = spk_model.pair_scaler.transform(sup_val["X_spk"])
    val_spk_probs = 0.65 * spk_model.pair_gbdt.predict_proba(val_spk_X)[:, 1] + 0.35 * spk_model.pair_logreg.predict_proba(val_spk_X)[:, 1]
    val_spk_auroc = float(roc_auc_score(sup_val["y_spk"], val_spk_probs))
    print(f"[Model 4] SpeakerAttributionModel Pairwise Val AUROC = {val_spk_auroc:.4f}")

    best_th = tune_story_threshold(story_clf, order_ranker, narr_clf, spk_model, train_ids, seq_map, labels, cache)
    print(f"Tuned Story-Text Probability Threshold on Train Split: {best_th:.2f}")

    print("\n--- Evaluating Baselines & Ablations on 16 Validation Sequences ---")
    val_gold = {sid: labels[sid] for sid in val_ids}
    experiments: Dict[str, Any] = {}

    preds_b0 = run_unmodified_fullpage_baseline(val_ids, seq_map)
    rep_b0 = evaluate(val_gold, preds_b0)
    experiments["01_unmodified_fullpage_ocr_baseline"] = rep_b0["macro"]

    preds_b1 = {
        sid: predict_sequence_with_pipeline(
            sid, seq_map[sid], cache, None, None, None, spk_model,
            filter_mode="heuristic", order_mode="heuristic", speaker_mode="single_speaker"
        )
        for sid in val_ids
    }
    rep_b1 = evaluate(val_gold, preds_b1)
    experiments["02_heuristic_bubbles_single_speaker"] = rep_b1["macro"]

    preds_a2 = {
        sid: predict_sequence_with_pipeline(
            sid, seq_map[sid], cache, story_clf, None, None, spk_model,
            filter_mode="learned", order_mode="heuristic", speaker_mode="single_speaker"
        )
        for sid in val_ids
    }
    rep_a2 = evaluate(val_gold, preds_a2)
    experiments["03_learned_story_classifier_single_speaker"] = rep_a2["macro"]

    preds_a3 = {
        sid: predict_sequence_with_pipeline(
            sid, seq_map[sid], cache, story_clf, order_ranker, None, spk_model,
            filter_mode="learned", order_mode="learned", speaker_mode="single_speaker"
        )
        for sid in val_ids
    }
    rep_a3 = evaluate(val_gold, preds_a3)
    experiments["04_learned_story_plus_learned_order_ranker"] = rep_a3["macro"]

    preds_a4 = {
        sid: predict_sequence_with_pipeline(
            sid, seq_map[sid], cache, story_clf, order_ranker, narr_clf, spk_model,
            filter_mode="learned", order_mode="learned", speaker_mode="spatial_kmeans"
        )
        for sid in val_ids
    }
    rep_a4 = evaluate(val_gold, preds_a4)
    experiments["05_plus_narration_and_spatial_clustering"] = rep_a4["macro"]

    preds_full = {
        sid: predict_sequence_with_pipeline(
            sid, seq_map[sid], cache, story_clf, order_ranker, narr_clf, spk_model,
            filter_mode="learned", order_mode="learned", speaker_mode="learned"
        )
        for sid in val_ids
    }
    rep_full = evaluate(val_gold, preds_full)
    experiments["06_full_adapted_pipeline"] = rep_full["macro"]

    save_predictions_jsonl(preds_full, val_ids, Path("experiments/val_predictions.jsonl"))
    Path("experiments/val_scores_full.json").write_text(json.dumps(rep_full, indent=2) + "\n", encoding="utf-8")

    for name, macro in experiments.items():
        print(
            f"{name:45s} | text_order={macro['text_order_score']:.4f} | "
            f"bal_joint_f1={macro['balanced_joint_f1']:.4f} | "
            f"joint_f1={macro['joint_f1']:.4f} | "
            f"spk_acc={macro['speaker_accuracy_on_matched']:.4f} | "
            f"tok_cov={macro['matched_token_coverage']:.4f}"
        )

    print("\n--- Running 5-Fold Cross-Validation across All 80 Development Sequences ---")
    oof_preds: Dict[str, List[List[Dict[str, str]]]] = {}
    for fold in range(5):
        f_val_ids = [dev_ids[i] for i in range(len(dev_ids)) if i % 5 == fold]
        f_tr_ids = [sid for sid in dev_ids if sid not in set(f_val_ids)]
        f_sup = build_supervision_for_sequences(f_tr_ids, seq_map, raw_labels, cache)

        f_story = StoryTextClassifier(seed=42 + fold)
        f_story.fit(f_sup["all_cands"], f_sup["y_story"])
        f_story.threshold = best_th
        f_story.empty_page_max_thresh = best_th + 0.06

        f_order = ReadingOrderRanker(seed=42 + fold)
        f_order.fit(f_sup["X_order"], f_sup["y_order"])

        f_narr = NarrationClassifier(seed=42 + fold)
        f_narr.fit(f_sup["pos_cands"], f_sup["y_narr"])

        f_spk = SpeakerAttributionModel(seed=42 + fold)
        f_spk.fit(f_sup["X_spk"], f_sup["y_spk"], f_sup["X_cnt"], f_sup["y_cnt"])

        for sid in f_val_ids:
            oof_preds[sid] = predict_sequence_with_pipeline(
                sid, seq_map[sid], cache, f_story, f_order, f_narr, f_spk,
                filter_mode="learned", order_mode="learned", speaker_mode="learned"
            )

    oof_path = Path("experiments/oof_predictions.jsonl")
    save_predictions_jsonl(oof_preds, dev_ids, oof_path)
    oof_report = evaluate(labels, load_jsonl(oof_path))
    Path("experiments/oof_scores.json").write_text(json.dumps(oof_report, indent=2) + "\n", encoding="utf-8")
    print("5-Fold Out-of-Fold Macro Scores (80 sequences):")
    print(json.dumps(oof_report["macro"], indent=2))

    print("\n--- Fitting Final Production Pipeline on All 80 Development Sequences ---")
    sup_all = build_supervision_for_sequences(dev_ids, seq_map, raw_labels, cache)
    final_story = StoryTextClassifier(seed=42)
    final_story.fit(sup_all["all_cands"], sup_all["y_story"])
    final_story.threshold = best_th
    final_story.empty_page_max_thresh = best_th + 0.06

    final_order = ReadingOrderRanker(seed=42)
    final_order.fit(sup_all["X_order"], sup_all["y_order"])

    final_narr = NarrationClassifier(seed=42)
    final_narr.fit(sup_all["pos_cands"], sup_all["y_narr"])

    final_spk = SpeakerAttributionModel(seed=42)
    final_spk.fit(sup_all["X_spk"], sup_all["y_spk"], sup_all["X_cnt"], sup_all["y_cnt"])

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    if final_story.mlp is not None:
        torch.save(final_story.mlp.state_dict(), models_dir / "story_mlp_weights.pt")
    with open(models_dir / "trained_pipeline.pkl", "wb") as f:
        pickle.dump(
            {
                "story_clf": final_story,
                "order_ranker": final_order,
                "narr_clf": final_narr,
                "spk_model": final_spk,
                "best_threshold": best_th,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    summary = {
        "split": {
            "train_sequences": len(train_ids),
            "val_sequences": len(val_ids),
            "train_pages": len(train_ids) * 3,
            "val_pages": len(val_ids) * 3,
        },
        "component_validation_metrics": {
            "story_text_classifier_val_auroc": val_story_auroc,
            "story_text_classifier_val_f1": val_story_f1,
            "reading_order_ranker_pairwise_val_acc": val_order_acc,
            "reading_order_ranker_pairwise_val_auroc": val_order_auroc,
            "narration_classifier_val_auroc": val_narr_auroc,
            "speaker_coref_pairwise_val_auroc": val_spk_auroc,
            "tuned_story_threshold": best_th,
        },
        "validation_ablations_16_sequences": experiments,
        "five_fold_cv_macro_80_sequences": oof_report["macro"],
        "runtime_seconds": round(time.time() - t_start, 2),
    }
    Path("experiments/experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved trained models to models/ and experiment logs to experiments/ in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
