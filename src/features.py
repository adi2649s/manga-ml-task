#!/usr/bin/env python3
"""
Multimodal candidate text region proposal, double-bubble lobe splitting,
panel segmentation, composite-strip Tesseract LSTM OCR, and feature extraction.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

TESSERACT_BIN = (
    os.environ.get("TESSERACT_BIN")
    or shutil.which("tesseract")
    or "/opt/homebrew/bin/tesseract"
)


def resolve_data_path(rel_path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve dataset paths portably whether files are located directly in the project
    directory (e.g. ./sequences.json) or inside a ./dataset/ subdirectory
    (e.g. ./dataset/sequences.json).
    """
    rel = Path(rel_path)
    if rel.is_absolute() and rel.exists():
        return rel
    roots = [base_dir] if base_dir is not None else []
    roots.extend([Path(__file__).resolve().parent.parent, Path.cwd()])
    for root in roots:
        for candidate in (root / rel, root / "dataset" / rel):
            if candidate.exists():
                return candidate
    return (roots[0] / rel) if roots else rel


COMMON_WORDS = {
    "a", "about", "actually", "after", "again", "all", "almost", "already", "also", "always",
    "am", "an", "and", "another", "any", "anyone", "anything", "anywhere", "are", "aren't",
    "around", "as", "ask", "asking", "at", "away", "back", "bad", "be", "because", "been",
    "before", "behind", "being", "believe", "best", "better", "between", "big", "bit", "blood",
    "body", "both", "boy", "boys", "bring", "brother", "but", "by", "call", "called", "came",
    "can", "can't", "cannot", "care", "case", "cause", "chance", "change", "child", "children",
    "class", "close", "club", "cold", "come", "coming", "completely", "cooking", "cool", "could",
    "couldn't", "course", "crap", "cut", "cute", "dad", "damn", "date", "dating", "day", "days",
    "dead", "deal", "death", "debt", "debts", "decided", "definitely", "devil", "devils", "did",
    "didn't", "die", "died", "different", "dinner", "do", "doctor", "does", "doesn't", "doing",
    "don't", "done", "doubt", "doubts", "down", "dream", "dreams", "drop", "dropping", "easy",
    "eat", "eating", "either", "else", "end", "enough", "even", "ever", "every", "everybody",
    "everyone", "everything", "exactly", "explain", "eyes", "face", "fact", "fake", "faked",
    "fall", "family", "far", "farm", "fast", "father", "feel", "feeling", "feels", "few",
    "fight", "fighting", "finally", "find", "fine", "finish", "first", "fish", "five", "floor",
    "food", "for", "found", "four", "fourth", "friend", "friends", "from", "front", "full",
    "fun", "future", "game", "games", "gave", "get", "getting", "girl", "girls", "give", "go",
    "god", "going", "gonna", "good", "got", "gotta", "grand", "grandpa", "great", "group",
    "grow", "guess", "guy", "guys", "ha", "had", "hair", "half", "hand", "hands", "happen",
    "happened", "happens", "happy", "hard", "has", "hate", "have", "having", "he", "he'd",
    "he'll", "he's", "head", "hear", "heard", "heart", "heck", "hell", "help", "helped", "helps",
    "her", "here", "hey", "hi", "high", "him", "himself", "his", "hold", "holding", "home",
    "honest", "honestly", "hope", "house", "how", "huh", "human", "humans", "hungry", "hunter",
    "hurry", "hurt", "i", "i'd", "i'll", "i'm", "i've", "idea", "if", "important", "in",
    "inside", "into", "invite", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "job",
    "just", "keep", "kid", "kidding", "kill", "killed", "kind", "knew", "know", "land", "last",
    "late", "later", "leave", "left", "let", "let's", "life", "like", "little", "live", "living",
    "loan", "long", "look", "looking", "looks", "lose", "loses", "lost", "lot", "love", "made",
    "make", "makes", "making", "man", "many", "matter", "may", "maybe", "me", "mean", "means",
    "meet", "men", "might", "mind", "minute", "mom", "money", "month", "more", "morning", "most",
    "mother", "mountains", "move", "movie", "movies", "much", "must", "my", "myself", "name",
    "need", "never", "new", "next", "nice", "night", "no", "nobody", "none", "normal", "normally",
    "not", "nothing", "now", "of", "off", "oh", "okay", "old", "on", "once", "one", "ones",
    "only", "open", "option", "or", "other", "others", "our", "out", "outside", "over", "owe",
    "own", "part", "pay", "paying", "people", "person", "pet", "piece", "pieces", "place",
    "play", "playing", "please", "point", "power", "practice", "president", "pretty", "probably",
    "problem", "problems", "put", "quite", "rain", "rather", "raw", "read", "ready", "real",
    "realize", "really", "reason", "refuse", "remember", "rest", "right", "room", "run",
    "running", "said", "sake", "same", "save", "saw", "say", "saying", "scared", "scary",
    "school", "second", "secret", "see", "seeing", "seem", "seems", "seen", "sell", "sense",
    "sent", "seriously", "she", "she's", "shit", "should", "shouldn't", "show", "sick", "side",
    "since", "single", "sister", "sisters", "situation", "skip", "skipping", "sleep", "slice",
    "small", "smile", "sneak", "so", "some", "someone", "something", "sometimes", "son", "soon",
    "sorry", "sort", "sound", "sounds", "speak", "start", "started", "stay", "still", "stop",
    "straight", "strong", "stubborn", "student", "students", "studies", "stuff", "stupid",
    "such", "sudden", "sugar", "sure", "take", "taken", "taking", "talk", "talked", "talking",
    "teacher", "teachers", "tell", "telling", "than", "thank", "thanks", "that", "that's",
    "the", "their", "them", "themselves", "then", "there", "there's", "these", "they", "they'll",
    "they're", "thing", "things", "think", "thinking", "third", "this", "those", "though",
    "thought", "thoughts", "three", "through", "time", "times", "to", "today", "together",
    "told", "tomorrow", "tonight", "too", "took", "top", "total", "touch", "tried", "trouble",
    "true", "truly", "trust", "truth", "try", "trying", "turn", "turns", "two", "uh", "um",
    "under", "understand", "until", "up", "us", "use", "used", "usual", "very", "vice",
    "vice-president", "visit", "voice", "wait", "waited", "walk", "wanna", "want", "wanted",
    "wants", "was", "wasn't", "watch", "watching", "water", "way", "we", "we'd", "we'll",
    "we're", "we've", "well", "went", "were", "weren't", "what", "what's", "whatever", "when",
    "where", "whether", "which", "while", "who", "who's", "whole", "why", "will", "win", "wish",
    "with", "without", "woman", "women", "won't", "wonder", "wonderful", "woof", "word", "words",
    "work", "worked", "working", "world", "worried", "worry", "worse", "would", "wouldn't",
    "wow", "wrong", "ya", "yeah", "year", "years", "yen", "yes", "yet", "you", "you'd", "you'll",
    "you're", "you've", "young", "your", "yourself", "ah", "eh", "hm", "hmm", "huh", "mhm",
    "mhmm", "ooh", "uh-huh", "woah", "whoa", "gaha", "eek", "ack", "ugh", "gah", "argh", "aaah",
    "aaaah", "nooo", "whaaat", "ehhh",
}

SFX_PATTERNS = re.compile(
    r"^(sfx\b|klack|clack|thud|thump|bam|wham|slam|boom|crash|rustle|rattl|creak|squeak|"
    r"fwip|fwoosh|whoosh|swish|slash|slice|stab|drip|drop|splash|splat|crackle|sizzle|"
    r"rumble|tremble|shiver|shudder|twitch|flinch|gasp|pant|puff|wheeze|cough|sniff|"
    r"snort|gulp|swallow|chew|munch|crunch|chomp|slurp|lick|kiss|smack|slap|punch|kick|"
    r"step|tap|pat|poke|grab|grip|squeeze|crush|rip|tear|snap|pop|click|clink|clang|"
    r"ding|ring|buzz|hum|beep|honk|screech|skid|zoom|vroo+m|roar|growl|howl|bark|meow|"
    r"chirp|flutter|flap|throb|ba-dump|badump|lub-dub|dokun|zawa|niko|gogo+|dodon|don\b|"
    r"plop|plip|chk|shk|ksh|fsh|zzt|bzt|glare|stare|glance|peek|blink|wink|nod|shake|"
    r"sigh|groan|moan|sob|sniffle|chuckle|giggle|snicker|smirk|grin|smile|blush|sweat)",
    re.IGNORECASE,
)


def detect_panels(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Recursive XY-cut panel segmentation using low-ink horizontal and vertical gutters."""
    H, W = img.shape
    dark = (img < 185).astype(np.uint8)
    row_dark = dark.mean(axis=1)
    col_dark = dark.mean(axis=0)
    r_idx = np.where(row_dark > 0.03)[0]
    c_idx = np.where(col_dark > 0.03)[0]
    if len(r_idx) < 20 or len(c_idx) < 20:
        return [(0, 0, W, H)]
    y0, y1 = int(r_idx.min()), int(r_idx.max()) + 1
    x0, x1 = int(c_idx.min()), int(c_idx.max()) + 1

    h_cuts = [y0]
    in_gutter = False
    g_start = y0
    for r in range(y0 + 50, y1 - 50):
        if dark[r, x0:x1].mean() < 0.018:
            if not in_gutter:
                in_gutter = True
                g_start = r
        else:
            if in_gutter:
                g_mid = (g_start + r) // 2
                if g_mid - h_cuts[-1] >= 80:
                    h_cuts.append(g_mid)
                in_gutter = False
    if y1 - h_cuts[-1] >= 80:
        h_cuts.append(y1)
    else:
        h_cuts[-1] = y1

    panels = []
    for bi in range(len(h_cuts) - 1):
        ya, yb = h_cuts[bi], h_cuts[bi + 1]
        band_dark = dark[ya:yb, x0:x1]
        col_d = band_dark.mean(axis=0)
        v_cuts = [x0]
        in_vg = False
        vg_start = x0
        for c in range(x0 + 50, x1 - 50):
            if col_d[c - x0] < 0.018:
                if not in_vg:
                    in_vg = True
                    vg_start = c
            else:
                if in_vg:
                    vg_mid = (vg_start + c) // 2
                    if vg_mid - v_cuts[-1] >= 70:
                        v_cuts.append(vg_mid)
                    in_vg = False
        if x1 - v_cuts[-1] >= 70:
            v_cuts.append(x1)
        else:
            v_cuts[-1] = x1
        sub_panels = []
        for vi in range(len(v_cuts) - 1):
            xa, xb = v_cuts[vi], v_cuts[vi + 1]
            sub_panels.append((xa, ya, xb - xa, yb - ya))
        sub_panels.sort(key=lambda p: -(p[0] + p[2] / 2.0))
        panels.extend(sub_panels)
    return panels


def detect_ellipsis_and_punct_in_holes(statsh: np.ndarray) -> Tuple[int, List[Tuple[float, float]]]:
    """Detect collinear 3-dot ellipses inside a bubble's hole components."""
    dots = []
    for j in range(1, len(statsh)):
        hx, hy, hw, hh, harea = statsh[j]
        if 2 <= hw <= 10 and 2 <= hh <= 10 and harea >= 3 and 0.35 <= (hw / float(hh)) <= 2.6:
            dots.append((hx + hw / 2.0, hy + hh / 2.0, hw, hh, j))
    dots.sort(key=lambda d: d[0])
    ellipses = []
    used = set()
    for a in range(len(dots)):
        if a in used:
            continue
        for b in range(a + 1, len(dots)):
            if b in used:
                continue
            for c in range(b + 1, len(dots)):
                if c in used:
                    continue
                d1, d2, d3 = dots[a], dots[b], dots[c]
                if abs(d1[1] - d2[1]) <= 4.5 and abs(d2[1] - d3[1]) <= 4.5 and abs(d1[1] - d3[1]) <= 5.5:
                    dx1 = d2[0] - d1[0]
                    dx2 = d3[0] - d2[0]
                    if 3.0 <= dx1 <= 22.0 and 3.0 <= dx2 <= 22.0 and abs(dx1 - dx2) <= 6.0:
                        ellipses.append(((d1[0] + d3[0]) / 2.0, (d1[1] + d3[1]) / 2.0))
                        used.update([a, b, c])
                        break
            if a in used:
                break
    return len(ellipses), ellipses


def cluster_glyphs_into_lobes(
    glyph_mask: np.ndarray,
    sub_img: np.ndarray,
    offset_x: int,
    offset_y: int,
    parent_id: str,
    source: str,
    bubble_meta: Dict[str, Any],
    filled_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Two-pass line-graph clustering to separate side-by-side lobes in double speech bubbles.
    For enclosed bubbles (filled_mask is not None), extracts the unmasked grayscale interior of
    filled_mask within each lobe's bounding box (masking only other lobes' line boxes) so that
    touching or edge characters are never clipped.
    """
    H_s, W_s = sub_img.shape
    # Use a compact (7, 2) horizontal kernel so words within a column connect, but side-by-side
    # lobes separated by an 8-12px vertical gap in double bubbles do not bridge!
    kw = 7 if filled_mask is not None else 10
    line_mask = cv2.dilate(glyph_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 2)))
    nl, ll, statsl, _ = cv2.connectedComponentsWithStats(line_mask)
    lines = []
    for l_idx in range(1, nl):
        lx, ly, lw, lh, larea = statsl[l_idx]
        if larea >= 6 and lh <= 58:
            lines.append((l_idx, int(lx), int(ly), int(lw), int(lh), int(larea)))
    if not lines:
        return []
    n = len(lines)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Pass 1: Vertical stacking of horizontally aligned lines
    for a in range(n):
        _, x1, y1, w1, h1, _ = lines[a]
        cx1 = x1 + w1 / 2.0
        for b in range(a + 1, n):
            _, x2, y2, w2, h2, _ = lines[b]
            cx2 = x2 + w2 / 2.0
            y_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            x_ov = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            if y_gap <= 22 and (x_ov / max(min(w1, w2), 1) >= 0.32 or abs(cx1 - cx2) <= 0.36 * max(w1, w2)):
                union(a, b)

    clusters: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {}
    for a in range(n):
        clusters.setdefault(find(a), []).append(lines[a])

    # Pass 2: Merge single-line horizontal segments on the exact same baseline (e.g. wide captions),
    # and attach tiny 1-line punctuation/short-word fragments inside a column to their vertically adjacent line
    c_keys = list(clusters.keys())
    if len(c_keys) > 1:
        for i_k in range(len(c_keys)):
            for j_k in range(i_k + 1, len(c_keys)):
                g1, g2 = clusters[c_keys[i_k]], clusters[c_keys[j_k]]
                if len(g1) == 1 and len(g2) == 1:
                    _, x1, y1, w1, h1, _ = g1[0]
                    _, x2, y2, w2, h2, _ = g2[0]
                    if abs((y1 + h1 / 2.0) - (y2 + h2 / 2.0)) <= 7 and max(0, max(x1, x2) - min(x1 + w1, x2 + w2)) <= 18:
                        union(lines.index(g1[0]), lines.index(g2[0]))
                elif len(g1) == 1 or len(g2) == 1:
                    small_g, big_g = (g1, g2) if len(g1) == 1 else (g2, g1)
                    _, sx, sy, sw, sh, _ = small_g[0]
                    bx0 = min(item[1] for item in big_g)
                    bx1 = max(item[1] + item[3] for item in big_g)
                    by0 = min(item[2] for item in big_g)
                    by1 = max(item[2] + item[4] for item in big_g)
                    x_ov_b = max(0, min(sx + sw, bx1) - max(sx, bx0))
                    y_gap_b = max(0, max(sy, by0) - min(sy + sh, by1))
                    if x_ov_b / max(sw, 1) >= 0.50 and y_gap_b <= 24:
                        union(lines.index(small_g[0]), lines.index(big_g[0]))
        clusters = {}
        for a in range(n):
            clusters.setdefault(find(a), []).append(lines[a])

    lobe_list = sorted(clusters.values(), key=lambda grp: -np.mean([g[1] + g[3] / 2.0 for g in grp]))
    out = []
    n_total_lobes = len(lobe_list)

    for idx_l, grp in enumerate(lobe_list):
        pad = 5
        gx0 = max(0, min(item[1] for item in grp) - pad)
        gy0 = max(0, min(item[2] for item in grp) - pad)
        gx1 = min(W_s, max(item[1] + item[3] for item in grp) + pad)
        gy1 = min(H_s, max(item[2] + item[4] for item in grp) + pad)
        gw, gh = gx1 - gx0, gy1 - gy0
        if gw < 6 or gh < 4:
            continue

        if filled_mask is not None:
            # Keep raw grayscale interior inside filled_mask so no anti-aliased or touching characters are clipped!
            crop = sub_img[gy0:gy1, gx0:gx1].copy()
            crop[filled_mask[gy0:gy1, gx0:gx1] == 0] = 255
            # Mask out line boxes belonging to OTHER lobes in this double/triple bubble
            for o_idx, o_grp in enumerate(lobe_list):
                if o_idx != idx_l:
                    for _, ox, oy, ow, oh, _ in o_grp:
                        ix0 = max(0, ox - 2 - gx0)
                        iy0 = max(0, oy - 2 - gy0)
                        ix1 = min(gw, ox + ow + 2 - gx0)
                        iy1 = min(gh, oy + oh + 2 - gy0)
                        if ix1 > ix0 and iy1 > iy0:
                            crop[iy0:iy1, ix0:ix1] = 255
            g_pix = int((glyph_mask[gy0:gy1, gx0:gx1] > 0).sum())
        else:
            m = np.zeros_like(glyph_mask)
            for l_idx, _, _, _, _, _ in grp:
                m[ll == l_idx] = glyph_mask[ll == l_idx]
            lobe_g = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            g_pix = int((m > 0).sum())
            if g_pix < 6:
                continue
            crop = np.full((gh, gw), 255, dtype=np.uint8)
            sub_lobe_g = lobe_g[gy0:gy1, gx0:gx1]
            crop[sub_lobe_g > 0] = sub_img[gy0:gy1, gx0:gx1][sub_lobe_g > 0]

        line_hs = [g[4] for g in grp]
        cand = {
            "bbox": (offset_x + gx0, offset_y + gy0, gw, gh),
            "bubble_bbox": bubble_meta.get("bubble_bbox", (offset_x + gx0, offset_y + gy0, gw, gh)),
            "crop": crop,
            "parent_id": parent_id,
            "source": source,
            "lobe_idx": idx_l,
            "n_lobes_in_bubble": n_total_lobes,
            "n_lines": len(grp),
            "med_line_h": float(np.median(line_hs)),
            "std_line_h": float(np.std(line_hs)) if len(line_hs) > 1 else 0.0,
            "glyph_pixel_count": max(g_pix, 10),
        }
        cand.update({k: v for k, v in bubble_meta.items() if k != "bubble_bbox"})
        out.append(cand)
    return out


def clean_ocr_text(raw_text: str, n_ellipses: int = 0, n_holes: int = 0) -> str:
    """Clean OCR output while preserving stutters, repeats, and meaningful punctuation."""
    txt = " ".join(raw_text.split())

    def _dehyphen(m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        if len(left) == 1:
            return f"{left}-{right}"
        if right.lower() in (
            "san", "kun", "chan", "sama", "senpai", "sensei", "president",
            "attack", "out", "up", "off", "in", "on", "two", "signer",
        ):
            return f"{left}-{right}"
        return f"{left}{right}"

    txt = re.sub(r"([A-Za-z]+)-\s+([A-Za-z]+)", _dehyphen, txt)
    txt = txt.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"').replace("…", "...")
    txt = re.sub(r"''+", "'", txt)
    txt = re.sub(r"\b([A-Za-z]+)/([TSREMLDtsremld])\b", r"\1'\2", txt)
    txt = re.sub(r"'\s*/\s*([TSREMLDtsremld])\b", r"'\1", txt)
    txt = re.sub(r"/\?", "!?", txt)
    txt = re.sub(r"!\s*7\b", "!?", txt)
    txt = re.sub(r"\b30\s+HARD\b", "SO HARD", txt)
    txt = re.sub(r"\b50\s+HARD\b", "SO HARD", txt)
    txt = re.sub(r"\.{3,}", "...", txt)
    txt = re.sub(r"\.\s+\.\s+\.", "...", txt)
    txt = re.sub(r"\.\s+\.", "..", txt)
    txt = re.sub(r"^[|\\/_~`^+=<>©®™§£¥¢°•·]+\s*", "", txt)
    txt = re.sub(r"\s*[|\\/_~`^+=<>©®™§£¥¢°•·]+$", "", txt)

    tokens = txt.split()
    cleaned_tokens = []
    for t in tokens:
        if t in ("|", "\\", "/", "_", "~", "^", "=", "+", "<", ">", "¢", "£", "¥", "©", "®", "™"):
            continue
        cleaned_tokens.append(t)
    txt = " ".join(cleaned_tokens)

    word_fixes = {
        "IMA": "I'M A",
        "IM": "I'M",
        "IvmM": "I'M",
        "TLL": "I'LL",
        "ILL": "I'LL",
        "IVE": "I'VE",
        "D0": "DO",
        "T0": "TO",
        "S0": "SO",
        "$0": "SO",
        "N0": "NO",
        "G0": "GO",
        "Y0U": "YOU",
        "DFFERENT": "DIFFERENT",
        "THiS": "THIS",
        "FARI": "FAR!",
    }
    out_words = []
    for w in txt.split():
        m = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9'$]+?)([^A-Za-z0-9]*)$", w)
        if m:
            pre, core, post = m.groups()
            if core in word_fixes:
                core = word_fixes[core]
            elif core.isupper() and len(core) >= 2:
                if any(c.isalpha() for c in core) and "0" in core and not core.isdigit():
                    core = core.replace("0", "O")
            out_words.append(f"{pre}{core}{post}")
        else:
            out_words.append(w)
    txt = " ".join(out_words).strip()

    if n_ellipses >= 1:
        if not txt and n_holes <= 5:
            txt = "..."
        elif txt and "..." not in txt and not txt.endswith((".", "!", "?")):
            txt = txt + "..."
    return txt


def compute_candidate_features(cand: Dict[str, Any], page_img: np.ndarray, H: int, W: int) -> np.ndarray:
    """Compute a 50-dimensional multimodal feature vector (geometric, topological, panel, visual, lexical)."""
    x, y, w, h = cand["bbox"]
    bx, by, bw, bh = cand["bubble_bbox"]
    cx, cy = (x + w / 2.0) / W, (y + h / 2.0) / H
    nw, nh = w / float(W), h / float(H)
    nbw, nbh = bw / float(W), bh / float(H)
    area_norm = (w * h) / float(W * H)
    b_area_norm = (bw * bh) / float(W * H)
    aspect = w / float(max(h, 1))
    b_aspect = bw / float(max(bh, 1))

    rect_extent = float(cand.get("rect_extent", 0.5))
    solidity = float(cand.get("solidity", 0.7))
    circularity = float(cand.get("circularity", 0.5))
    is_enclosed = 1.0 if cand.get("source") == "enclosed" else 0.0
    is_inverted = 1.0 if cand.get("source") == "inverted" else 0.0
    n_lobes = float(cand.get("n_lobes_in_bubble", 1))
    lobe_idx = float(cand.get("lobe_idx", 0))
    n_lines = float(cand.get("n_lines", 1))
    med_line_h = float(cand.get("med_line_h", 14.0)) / 30.0
    std_line_h = float(cand.get("std_line_h", 0.0)) / 15.0
    n_holes = float(cand.get("n_holes", 5))
    n_ellipses = float(cand.get("n_ellipses", 0))
    glyph_density = float(cand.get("glyph_pixel_count", 50)) / float(max(w * h, 1))

    near_top = 1.0 if y < 0.045 * H else 0.0
    near_bottom = 1.0 if (y + h) > 0.92 * H else 0.0
    near_left = 1.0 if x < 0.04 * W else 0.0
    near_right = 1.0 if (x + w) > 0.96 * W else 0.0

    px_rel = float(cand.get("px_rel", cx))
    py_rel = float(cand.get("py_rel", cy))
    panel_area_ratio = float(cand.get("panel_area_ratio", 0.25))

    pad_b = 3
    x0_b, y0_b = max(0, bx - pad_b), max(0, by - pad_b)
    x1_b, y1_b = min(W, bx + bw + pad_b), min(H, by + bh + pad_b)
    b_patch = page_img[y0_b:y1_b, x0_b:x1_b]
    if b_patch.size > 0:
        top_edge_dark = float((b_patch[:min(5, b_patch.shape[0]), :] < 120).mean())
        bot_edge_dark = float((b_patch[max(0, b_patch.shape[0] - 5):, :] < 120).mean())
        left_edge_dark = float((b_patch[:, :min(5, b_patch.shape[1])] < 120).mean())
        right_edge_dark = float((b_patch[:, max(0, b_patch.shape[1] - 5):] < 120).mean())
        rect_border_score = (top_edge_dark + bot_edge_dark + left_edge_dark + right_edge_dark) / 4.0
        interior_white = float((b_patch > 225).mean())
    else:
        rect_border_score = 0.0
        interior_white = 0.5

    txt = cand.get("text", "")
    conf = float(cand.get("conf", 0.0)) / 100.0
    min_conf = float(cand.get("min_conf", 0.0)) / 100.0
    max_conf = float(cand.get("max_conf", 0.0)) / 100.0
    n_chars = len(txt)
    words = re.findall(r"[A-Za-z']+", txt)
    n_words = len(words)
    alpha_count = sum(c.isalpha() for c in txt)
    digit_count = sum(c.isdigit() for c in txt)
    punct_count = sum(not c.isalnum() and not c.isspace() for c in txt)
    upper_count = sum(c.isupper() for c in txt)
    vowels = sum(c.lower() in "aeiou" for c in txt)

    alpha_ratio = alpha_count / float(max(n_chars, 1))
    digit_ratio = digit_count / float(max(n_chars, 1))
    punct_ratio = punct_count / float(max(n_chars, 1))
    upper_ratio = upper_count / float(max(alpha_count, 1))
    vowel_ratio = vowels / float(max(alpha_count, 1))

    dict_matches = sum(w.lower().strip("'") in COMMON_WORDS for w in words)
    dict_ratio = dict_matches / float(max(n_words, 1))

    is_pure_punct = 1.0 if (n_chars > 0 and alpha_count == 0 and digit_count == 0 and any(p in txt for p in ("...", "!", "?"))) else 0.0
    is_page_num = 1.0 if (re.match(r"^\d{1,3}$", txt.strip()) and (near_bottom > 0 or near_top > 0 or near_left > 0 or near_right > 0)) else 0.0
    is_sfx = 1.0 if (SFX_PATTERNS.search(txt.strip()) is not None or "sfx:" in txt.lower()) else 0.0
    has_q_or_excl = 1.0 if ("?" in txt or "!" in txt) else 0.0
    has_quotes = 1.0 if ('"' in txt) else 0.0
    has_first_person = 1.0 if re.search(r"\b(i|i'm|i'll|i'd|i've|my|me|mine|we|we're|we'll|our|us)\b", txt, re.I) else 0.0
    has_second_person = 1.0 if re.search(r"\b(you|you're|you'll|your|yours|yourself)\b", txt, re.I) else 0.0

    feats = np.array([
        cx, cy, nw, nh, nbw, nbh, area_norm, b_area_norm, aspect, b_aspect,
        rect_extent, solidity, circularity, is_enclosed, is_inverted,
        n_lobes / 4.0, lobe_idx / 4.0, n_lines / 10.0, med_line_h, std_line_h,
        min(n_holes, 100.0) / 100.0, min(n_ellipses, 3.0) / 3.0, glyph_density,
        near_top, near_bottom, near_left, near_right,
        px_rel, py_rel, panel_area_ratio,
        rect_border_score, interior_white,
        conf, min_conf, max_conf,
        min(n_chars, 200) / 200.0, min(n_words, 40) / 40.0,
        alpha_ratio, digit_ratio, punct_ratio, upper_ratio, vowel_ratio,
        dict_ratio, is_pure_punct, is_page_num, is_sfx,
        has_q_or_excl, has_quotes, has_first_person, has_second_person,
    ], dtype=np.float32)
    return feats


def extract_visual_context_descriptor(page_img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    H, W = page_img.shape
    x, y, w, h = bbox
    pad_x, pad_y = max(30, int(w * 0.45)), max(30, int(h * 0.45))
    patches = [
        page_img[max(0, y - pad_y):max(1, y), max(0, x):min(W, x + w)],
        page_img[min(H - 1, y + h):min(H, y + h + pad_y), max(0, x):min(W, x + w)],
        page_img[max(0, y):min(H, y + h), max(0, x - pad_x):max(1, x)],
        page_img[max(0, y):min(H, y + h), min(W - 1, x + w):min(W, x + w + pad_x)],
    ]
    desc = []
    for p in patches:
        if p.size > 10:
            p_f = p.astype(np.float32) / 255.0
            desc.extend([
                float(p_f.mean()),
                float(p_f.std()),
                float((p < 80).mean()),
                float((p > 230).mean()),
                float(((p >= 80) & (p <= 200)).mean()),
            ])
        else:
            desc.extend([1.0, 0.0, 0.0, 1.0, 0.0])
    desc.extend([(x + w / 2.0) / W, (y + h / 2.0) / H, w / float(W), h / float(H)])
    return np.array(desc, dtype=np.float32)


def extract_page_candidates(img_path: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Full candidate extraction + OCR + feature computation for a single manga page."""
    resolved_img = resolve_data_path(img_path, base_dir)
    img = cv2.imread(str(resolved_img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path} (resolved: {resolved_img})")
    H, W = img.shape
    panels = detect_panels(img)

    covered_dark = np.zeros_like(img)
    candidates: List[Dict[str, Any]] = []

    # Pass 1: Enclosed White Bubbles (dual thresholds 230 and 218)
    for thresh_idx, w_thresh in enumerate([230, 218]):
        _, white = cv2.threshold(img, w_thresh, 255, cv2.THRESH_BINARY)
        num_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
        for i in range(1, num_w):
            x, y, w, h, area = [int(v) for v in stats_w[i]]
            if area < 240 or area > H * W * 0.45:
                continue
            if w < 14 or h < 12 or w > W * 0.95 or h > H * 0.95:
                continue
            if thresh_idx > 0 and (covered_dark[y:y + h, x:x + w] > 0).sum() > 25:
                continue

            comp_mask = (labels_w == i)[y:y + h, x:x + w].astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            main_contour = max(contours, key=cv2.contourArea)
            c_area = max(float(cv2.contourArea(main_contour)), 1.0)
            c_perim = max(float(cv2.arcLength(main_contour, True)), 1.0)
            hull = cv2.convexHull(main_contour)
            hull_area = max(float(cv2.contourArea(hull)), 1.0)
            solidity = min(1.0, c_area / hull_area)
            circularity = min(1.0, (4.0 * np.pi * c_area) / (c_perim * c_perim))
            rect_extent = min(1.0, float(area) / max(w * h, 1))

            filled = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(filled, contours, -1, 255, -1)
            sub_img = img[y:y + h, x:x + w]
            holes = ((filled > 0) & (comp_mask == 0) & (sub_img < 212)).astype(np.uint8) * 255

            nh, lh, statsh, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
            n_ellipses, _ = detect_ellipsis_and_punct_in_holes(statsh)

            glyph_holes = np.zeros_like(holes)
            valid_hole_count = 0
            for j in range(1, nh):
                hx, hy, hw, hh, harea = [int(v) for v in statsh[j]]
                # Allow connected multi-letter components up to hw <= 110 so touching letters are never dropped!
                if 2 <= hh <= 54 and 1 <= hw <= 110 and 2 <= harea <= 2400:
                    if (harea / float(hw * hh)) >= 0.06:
                        p = 3
                        x0, y0, x1, y1 = max(0, hx - p), max(0, hy - p), min(w, hx + hw + p), min(h, hy + hh + p)
                        ring_w = comp_mask[y0:y1, x0:x1]
                        ring_d = (lh[y0:y1, x0:x1] == j)
                        non_s = ring_w[~ring_d]
                        if len(non_s) > 0 and (non_s > 0).mean() >= 0.45:
                            glyph_holes[lh == j] = 255
                            valid_hole_count += 1

            if (glyph_holes > 0).sum() < 6 and n_ellipses == 0:
                continue

            covered_dark[y:y + h, x:x + w] = cv2.bitwise_or(
                covered_dark[y:y + h, x:x + w],
                cv2.dilate(glyph_holes, np.ones((5, 5), np.uint8)),
            )
            b_meta = {
                "bubble_bbox": (x, y, w, h),
                "rect_extent": rect_extent,
                "solidity": solidity,
                "circularity": circularity,
                "n_holes": valid_hole_count,
                "n_ellipses": n_ellipses,
            }
            lobes = cluster_glyphs_into_lobes(
                glyph_holes, sub_img, x, y, f"b_{thresh_idx}_{i}", "enclosed", b_meta, filled_mask=filled
            )
            candidates.extend(lobes)

    # Pass 2: Open-border / white-surround dark glyphs
    _, dark = cv2.threshold(img, 192, 255, cv2.THRESH_BINARY_INV)
    _, white_bg = cv2.threshold(img, 224, 255, cv2.THRESH_BINARY)
    dark_rem = cv2.bitwise_and(dark, cv2.bitwise_not(covered_dark))
    nc, lc, statsc, _ = cv2.connectedComponentsWithStats(dark_rem, connectivity=8)
    rem_glyphs = np.zeros_like(dark_rem)
    for i in range(1, nc):
        x, y, w, h, area = [int(v) for v in statsc[i]]
        if 3 <= h <= 46 and 1 <= w <= 80 and 3 <= area <= 1600:
            if (area / float(w * h)) >= 0.08:
                pad = 4
                x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
                bw_box = white_bg[y0:y1, x0:x1]
                bd_box = (lc[y0:y1, x0:x1] == i)
                non_self = bw_box[~bd_box]
                if len(non_self) > 0 and (non_self > 0).mean() >= 0.68:
                    rem_glyphs[lc == i] = 255

    open_meta = {
        "rect_extent": 0.55,
        "solidity": 0.65,
        "circularity": 0.45,
        "n_holes": 6,
        "n_ellipses": 0,
    }
    lobes_open = cluster_glyphs_into_lobes(rem_glyphs, img, 0, 0, "open", "open", open_meta, filled_mask=None)
    for idx_o, c in enumerate(lobes_open):
        if c["bbox"][2] >= 12 and c["bbox"][3] >= 6:
            c["parent_id"] = f"open_{idx_o}"
            c["bubble_bbox"] = c["bbox"]
            candidates.append(c)

    # Pass 3: White-on-Black speech/narration bubbles
    inv = 255 - img
    _, w_glyphs = cv2.threshold(img, 212, 255, cv2.THRESH_BINARY)
    _, d_bg = cv2.threshold(img, 48, 255, cv2.THRESH_BINARY_INV)
    nw_g, lw_g, st_g, _ = cv2.connectedComponentsWithStats(w_glyphs, connectivity=8)
    valid_w = np.zeros_like(w_glyphs)
    for i in range(1, nw_g):
        x, y, w, h, area = [int(v) for v in st_g[i]]
        if 5 <= h <= 40 and 1 <= w <= 60 and 5 <= area <= 1100:
            if (area / float(w * h)) >= 0.10:
                pad = 4
                x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
                bg_box = d_bg[y0:y1, x0:x1]
                fg_box = (lw_g[y0:y1, x0:x1] == i)
                non_s = bg_box[~fg_box]
                if len(non_s) > 0 and (non_s > 0).mean() >= 0.76:
                    valid_w[lw_g == i] = 255
    inv_meta = {
        "rect_extent": 0.65,
        "solidity": 0.70,
        "circularity": 0.50,
        "n_holes": 8,
        "n_ellipses": 0,
    }
    lobes_inv = cluster_glyphs_into_lobes(valid_w, inv, 0, 0, "inv", "inverted", inv_meta, filled_mask=None)
    for idx_i, c in enumerate(lobes_inv):
        if c["bbox"][2] >= 22 and c["bbox"][3] >= 12 and c["n_lines"] >= 1:
            bx, by, bw, bh = c["bbox"]
            x0, y0, x1, y1 = max(0, bx - 4), max(0, by - 4), min(W, bx + bw + 4), min(H, by + bh + 4)
            c["crop"] = inv[y0:y1, x0:x1].copy()
            c["parent_id"] = f"inv_{idx_i}"
            c["bubble_bbox"] = c["bbox"]
            candidates.append(c)

    for c in candidates:
        x, y, w, h = c["bbox"]
        cx, cy = x + w / 2.0, y + h / 2.0
        best_p = 0
        best_dist = float("inf")
        for p_idx, (px, py, pw, ph) in enumerate(panels):
            if px <= cx <= px + pw and py <= cy <= py + ph:
                best_p = p_idx
                break
            d = max(0.0, px - cx, cx - (px + pw)) + max(0.0, py - cy, cy - (py + ph))
            if d < best_dist:
                best_dist = d
                best_p = p_idx
        px, py, pw, ph = panels[best_p]
        c["panel_idx"] = best_p
        c["panel_bbox"] = (px, py, pw, ph)
        c["px_rel"] = float((cx - px) / max(pw, 1))
        c["py_rel"] = float((cy - py) / max(ph, 1))
        c["panel_area_ratio"] = float((pw * ph) / max(W * H, 1))

    candidates.sort(key=lambda c: (0 if c["source"] == "enclosed" else 1, -c["bbox"][2] * c["bbox"][3]))
    deduped = []
    for c in candidates:
        x1, y1, w1, h1 = c["bbox"]
        keep = True
        for d in deduped:
            x2, y2, w2, h2 = d["bbox"]
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            if inter > 0:
                min_a = min(w1 * h1, w2 * h2)
                if inter / float(max(min_a, 1)) > 0.70:
                    keep = False
                    break
        if keep:
            deduped.append(c)
    candidates = deduped

    if candidates:
        crops = []
        for c in candidates:
            c2 = cv2.resize(c["crop"], (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            c2 = cv2.copyMakeBorder(c2, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            crops.append(c2)
        max_w = max(c.shape[1] for c in crops) + 40
        total_h = sum(c.shape[0] + 60 for c in crops) + 40
        canvas = np.full((total_h, max_w), 255, dtype=np.uint8)
        bands = []
        cur_y = 20
        for c in crops:
            ch, cw = c.shape
            canvas[cur_y:cur_y + ch, 20:20 + cw] = c
            bands.append((cur_y, cur_y + ch))
            cur_y += ch + 60

        png = cv2.imencode(".png", canvas)[1].tobytes()
        res = subprocess.run(
            [TESSERACT_BIN, "stdin", "stdout", "--psm", "6", "tsv"],
            input=png,
            capture_output=True,
        )
        for c in candidates:
            c["words"] = []
            c["confs"] = []
        for line in res.stdout.decode("utf-8", "replace").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 12 and parts[0] == "5":
                word = parts[11].strip()
                if word:
                    yc = int(parts[7]) + int(parts[9]) / 2.0
                    for idx, (y0, y1) in enumerate(bands):
                        if y0 - 10 <= yc <= y1 + 10:
                            candidates[idx]["words"].append(word)
                            candidates[idx]["confs"].append(float(parts[10]))
                            break

    final_candidates = []
    for c in candidates:
        raw_txt = " ".join(c.get("words", []))
        confs = c.get("confs", [])
        c["raw_text"] = raw_txt
        c["conf"] = float(np.mean(confs)) if confs else (75.0 if c.get("n_ellipses", 0) >= 1 and c.get("n_holes", 0) <= 5 else 0.0)
        c["min_conf"] = float(np.min(confs)) if confs else c["conf"]
        c["max_conf"] = float(np.max(confs)) if confs else c["conf"]
        c["text"] = clean_ocr_text(raw_txt, c.get("n_ellipses", 0), c.get("n_holes", 0))
        if not c["text"].strip():
            continue
        del c["crop"]
        c["features"] = compute_candidate_features(c, img, H, W)
        c["visual_context"] = extract_visual_context_descriptor(img, c["bubble_bbox"])
        final_candidates.append(c)

    return {
        "img_path": img_path,
        "width": W,
        "height": H,
        "panels": panels,
        "candidates": final_candidates,
    }


def build_or_load_cache(
    sequences_path: str = "sequences.json",
    cache_path: str = "cache/extracted_pages.pkl",
    max_workers: int = 8,
    force: bool = False,
    base_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    root = base_dir or Path(__file__).resolve().parent.parent
    cache_file = root / cache_path if not Path(cache_path).is_absolute() else Path(cache_path)
    if cache_file.exists() and not force:
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    seq_file = resolve_data_path(sequences_path, root)
    seqs = json.loads(seq_file.read_text(encoding="utf-8"))
    all_pages = [p for s in seqs for p in s["images"]]
    t0 = time.time()
    results: Dict[str, Dict[str, Any]] = {}

    def _work(p: str) -> Tuple[str, Dict[str, Any]]:
        return p, extract_page_candidates(p, root)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, (p, data) in enumerate(ex.map(_work, all_pages), 1):
            results[p] = data
            if idx % 40 == 0 or idx == len(all_pages):
                print(f"[Cache] Processed {idx}/{len(all_pages)} pages ({time.time() - t0:.1f}s)")

    with open(cache_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return results


if __name__ == "__main__":
    build_or_load_cache(force=True)
