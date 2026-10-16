#!/usr/bin/env python3
"""
Standalone inference script that loads the trained open-weight models from models/
and generates test predictions in the required JSONL format (test_predictions.jsonl
and submission.jsonl) for the 15 test sequences in sample_submission.jsonl.
Works both with a pre-built cache and in a clean environment directly from test/images/*.png.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

from score import load_jsonl
from src.features import extract_page_candidates, resolve_data_path
from train_and_evaluate import predict_sequence_with_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent


def generate_test_predictions(
    sequences_path: Path = Path("sequences.json"),
    sample_submission_path: Path = Path("sample_submission.jsonl"),
    model_path: Path = Path("models/trained_pipeline.pkl"),
    cache_path: Path = Path("cache/extracted_pages.pkl"),
    output_paths: List[Path] = [Path("test_predictions.jsonl"), Path("submission.jsonl")],
    use_cache: bool = True,
) -> Dict[str, List[List[Dict[str, str]]]]:
    t0 = time.time()
    resolved_model = model_path if model_path.is_absolute() else (PROJECT_ROOT / model_path)
    with open(resolved_model, "rb") as f:
        bundle = pickle.load(f)

    story_clf = bundle["story_clf"]
    order_ranker = bundle["order_ranker"]
    narr_clf = bundle["narr_clf"]
    spk_model = bundle["spk_model"]
    story_clf.threshold = float(bundle.get("best_threshold", 0.54))
    story_clf.empty_page_max_thresh = story_clf.threshold + 0.06

    seq_file = resolve_data_path(sequences_path, PROJECT_ROOT)
    seqs = json.loads(seq_file.read_text(encoding="utf-8"))
    seq_map = {s["sequence_id"]: s for s in seqs}

    sub_file = resolve_data_path(sample_submission_path, PROJECT_ROOT)
    sample_rows = [
        json.loads(line)
        for line in sub_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_seq_ids = [r["sequence_id"] for r in sample_rows]

    resolved_cache = cache_path if cache_path.is_absolute() else (PROJECT_ROOT / cache_path)
    if use_cache and resolved_cache.exists():
        with open(resolved_cache, "rb") as f:
            cache: Dict[str, Dict[str, Any]] = pickle.load(f)
    else:
        cache = {}

    missing_test_pages = [
        img_path
        for sid in test_seq_ids
        for img_path in seq_map[sid]["images"]
        if img_path not in cache
    ]
    if missing_test_pages:
        def _extract(p: str) -> Tuple[str, Dict[str, Any]]:
            return p, extract_page_candidates(p, PROJECT_ROOT)

        with ThreadPoolExecutor(max_workers=8) as ex:
            for p, data in ex.map(_extract, missing_test_pages):
                cache[p] = data

    predictions: Dict[str, List[List[Dict[str, str]]]] = {}
    out_lines: List[str] = []
    total_lines = 0
    empty_pages = 0

    for sid in test_seq_ids:
        pages_pred = predict_sequence_with_pipeline(
            sid,
            seq_map[sid],
            cache,
            story_clf,
            order_ranker,
            narr_clf,
            spk_model,
            filter_mode="learned",
            order_mode="learned",
            speaker_mode="learned",
        )
        predictions[sid] = pages_pred
        for p in pages_pred:
            if not p:
                empty_pages += 1
            total_lines += len(p)
        out_lines.append(json.dumps({"sequence_id": sid, "pages": pages_pred}, ensure_ascii=False))

    jsonl_text = "\n".join(out_lines) + "\n"
    saved_paths = []
    for out_p in output_paths:
        target_p = out_p if out_p.is_absolute() else (PROJECT_ROOT / out_p)
        target_p.write_text(jsonl_text, encoding="utf-8")
        validated = load_jsonl(target_p, pages=3)
        assert list(validated.keys()) == test_seq_ids, f"Sequence ID mismatch in {target_p}"
        saved_paths.append(str(target_p.name))

    print(
        f"Generated & validated predictions for {len(test_seq_ids)} test sequences "
        f"({len(test_seq_ids)*3} pages, {total_lines} story lines, {empty_pages} empty pages) "
        f"in {time.time() - t0:.2f}s -> {saved_paths}"
    )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, default=Path("sequences.json"))
    parser.add_argument("--sample-submission", type=Path, default=Path("sample_submission.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("models/trained_pipeline.pkl"))
    parser.add_argument("--output", type=Path, default=Path("test_predictions.jsonl"))
    parser.add_argument("--no-cache", action="store_true", help="Run OCR directly on test PNGs ignoring cache")
    args = parser.parse_args()
    out_list = [args.output]
    if args.output.name != "submission.jsonl":
        out_list.append(Path("submission.jsonl"))
    generate_test_predictions(
        sequences_path=args.sequences,
        sample_submission_path=args.sample_submission,
        model_path=args.model,
        output_paths=out_list,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
