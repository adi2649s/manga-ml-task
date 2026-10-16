#!/usr/bin/env python3
"""
Learned components for The Manga Task [Advanced]:
1. StoryTextClassifier: PyTorch Residual MLP + HistGradientBoosting + RandomForest ensemble
   with character n-gram TF-IDF features for story-text vs. non-story region classification.
2. ReadingOrderRanker: Pairwise Gradient-Boosted Ranker + Exact Bitmask DP (Minimum Feedback
   Arc Set) tournament decoder for panel-aware right-to-left manga reading order.
3. NarrationClassifier: Multimodal (box geometry + lexical TF-IDF) classifier for NARRATION.
4. SpeakerAttributionModel: Pairwise Co-Reference Link Predictor + Sequence Speaker-Count
   Estimator + Constrained Agglomerative Clustering across 3-page sequences.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from score import TOKEN, align_tokens, distance, norm


def align_candidate_to_gold(c_text: str, gold_lines: List[Dict[str, str]]) -> Tuple[float, int]:
    """
    Align a candidate region's OCR text against the ground-truth lines of a page.
    Returns (match_score, best_gold_line_index).
    """
    c_norm = norm(c_text)
    c_toks = [(t, "x") for t in TOKEN.findall(c_norm)]
    if not c_toks:
        return 0.0, -1
    best_sc = 0.0
    best_g = -1
    for g_idx, g in enumerate(gold_lines):
        g_norm = norm(g["text"])
        g_toks = [(t, "x") for t in TOKEN.findall(g_norm)]
        if not g_toks:
            continue
        if c_norm == g_norm:
            return 1.0, g_idx
        char_sim = 1.0 - distance(c_norm, g_norm) / max(len(c_norm), len(g_norm), 1)
        pairs = align_tokens(g_toks, c_toks)
        w_sum = sum(w for _, _, w in pairs)
        prec = w_sum / len(c_toks)
        rec = w_sum / len(g_toks)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        has_long_match = any(len(gt[0]) >= 4 and w >= 0.70 for gt, _, w in pairs)
        if (w_sum >= 1.8 or has_long_match) and prec >= 0.55 and rec >= 0.18:
            sc = max(f1, 0.72 * prec + 0.28 * rec, char_sim)
        else:
            sc = max(f1, char_sim)
        if sc > best_sc:
            best_sc = sc
            best_g = g_idx
    return best_sc, best_g


class ResidualMLP(nn.Module):
    """Open-weight PyTorch neural network for multimodal region classification."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.GELU()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.act(h + self.block1(h))
        return self.head(h).squeeze(-1)


class StoryTextClassifier:
    """Multimodal classifier separating valid story text from SFX, page numbers, signs, and art noise."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=300, sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=24, random_state=seed)
        self.gbdt = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=15, random_state=seed
        )
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=1
        )
        self.mlp: Optional[ResidualMLP] = None
        self.threshold: float = 0.54
        self.empty_page_max_thresh: float = 0.60

    def _build_X(self, candidates: List[Dict[str, Any]], fit: bool = False) -> np.ndarray:
        dense = np.vstack([c["features"] for c in candidates])
        texts = [c.get("text", "") for c in candidates]
        if fit:
            tfidf_mat = self.tfidf.fit_transform(texts)
            n_comp = min(24, max(1, tfidf_mat.shape[1] - 1), max(1, tfidf_mat.shape[0] - 1))
            self.svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
            txt_svd = self.svd.fit_transform(tfidf_mat)
            X = np.hstack([dense, txt_svd])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return self.scaler.fit_transform(X).astype(np.float32)
        else:
            tfidf_mat = self.tfidf.transform(texts)
            txt_svd = self.svd.transform(tfidf_mat)
            X = np.hstack([dense, txt_svd])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return self.scaler.transform(X).astype(np.float32)

    def fit(self, train_candidates: List[Dict[str, Any]], y: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = self._build_X(train_candidates, fit=True)
        self.gbdt.fit(X, y)
        self.rf.fit(X, y)

        in_dim = X.shape[1]
        self.mlp = ResidualMLP(in_dim=in_dim, hidden_dim=128, dropout=0.20)
        pos_weight = torch.tensor([max(1.0, float((y == 0).sum()) / max(float((y == 1).sum()), 1.0))], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=1.5e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y.astype(np.float32))
        batch_size = 128
        n = len(X)
        self.mlp.train()
        for _ in range(35):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                logits = self.mlp(X_t[idx])
                loss = criterion(logits, y_t[idx])
                loss.backward()
                optimizer.step()
            scheduler.step()
        self.mlp.eval()

    def predict_proba(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        if not candidates:
            return np.zeros(0, dtype=np.float32)
        X = self._build_X(candidates, fit=False)
        p_gbdt = self.gbdt.predict_proba(X)[:, 1]
        p_rf = self.rf.predict_proba(X)[:, 1]
        if self.mlp is not None:
            self.mlp.eval()
            with torch.no_grad():
                logits = self.mlp(torch.from_numpy(X))
                p_mlp = torch.sigmoid(logits).numpy()
            probs = 0.45 * p_gbdt + 0.25 * p_rf + 0.30 * p_mlp
        else:
            probs = 0.65 * p_gbdt + 0.35 * p_rf

        for i, c in enumerate(candidates):
            feats = c["features"]
            is_page_num = feats[44] > 0.5
            is_sfx = feats[45] > 0.5
            if is_page_num or "sfx:" in c.get("text", "").lower():
                probs[i] *= 0.05
            elif is_sfx and c.get("source") != "enclosed":
                probs[i] *= 0.25
        return probs.astype(np.float32)


def pairwise_order_features(ci: Dict[str, Any], cj: Dict[str, Any], W: int, H: int) -> np.ndarray:
    """Features predicting whether bubble ci comes BEFORE bubble cj in manga reading order."""
    xi, yi, wi, hi = ci["bbox"]
    xj, yj, wj, hj = cj["bbox"]
    cxi, cyi = (xi + wi / 2.0) / W, (yi + hi / 2.0) / H
    cxj, cyj = (xj + wj / 2.0) / W, (yj + hj / 2.0) / H

    pi, pj = ci.get("panel_idx", 0), cj.get("panel_idx", 0)
    same_panel = 1.0 if pi == pj else 0.0
    panel_diff = float(pj - pi)

    same_parent = 1.0 if (ci.get("parent_id") == cj.get("parent_id") and ci.get("source") == "enclosed") else 0.0
    lobe_diff = float(cj.get("lobe_idx", 0) - ci.get("lobe_idx", 0)) if same_parent else 0.0

    dx = cxi - cxj
    dy = cyj - cyi
    top_diff = (yj - yi) / float(H)
    right_diff = ((xi + wi) - (xj + wj)) / float(W)
    bottom_above_top = ((yj - (yi + hi)) / float(H))

    x_ov = max(0, min(xi + wi, xj + wj) - max(xi, xj)) / float(max(min(wi, wj), 1))
    y_ov = max(0, min(yi + hi, yj + hj) - max(yi, yj)) / float(max(min(hi, hj), 1))

    px_diff = float(ci.get("px_rel", cxi) - cj.get("px_rel", cxj))
    py_diff = float(cj.get("py_rel", cyi) - ci.get("py_rel", cyj))

    heur_diff = ((cyj - 0.35 * cxj) - (cyi - 0.35 * cxi))
    angle = math.atan2(cyj - cyi, cxi - cxj) / math.pi

    return np.array([
        same_panel, panel_diff, same_parent, lobe_diff,
        dx, dy, top_diff, right_diff, bottom_above_top,
        x_ov, y_ov, px_diff, py_diff, heur_diff, angle,
        dx * same_panel, dy * same_panel, bottom_above_top * same_panel,
        dx * y_ov, dy * x_ov,
    ], dtype=np.float32)


class ReadingOrderRanker:
    """Pairwise ranking model + exact bitmask DP tournament solver for reading order."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.scaler = StandardScaler()
        self.gbdt = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=seed
        )
        self.logreg = LogisticRegression(C=1.0, random_state=seed)
        self.is_fitted = False

    def fit(self, X_pairs: np.ndarray, y_pairs: np.ndarray) -> None:
        if len(X_pairs) < 10:
            return
        Xs = self.scaler.fit_transform(X_pairs)
        self.gbdt.fit(Xs, y_pairs)
        self.logreg.fit(Xs, y_pairs)
        self.is_fitted = True

    def order_candidates(self, candidates: List[Dict[str, Any]], W: int, H: int) -> List[Dict[str, Any]]:
        grouped: List[List[Dict[str, Any]]] = []
        parent_map: Dict[str, int] = {}
        for c in sorted(candidates, key=lambda x: x.get("lobe_idx", 0)):
            pid = c.get("parent_id", "")
            if c.get("source") == "enclosed" and pid in parent_map:
                grouped[parent_map[pid]].append(c)
            else:
                if c.get("source") == "enclosed" and pid:
                    parent_map[pid] = len(grouped)
                grouped.append([c])

        reps = [grp[0] for grp in grouped]
        m = len(reps)
        if m <= 1:
            return [c for grp in grouped for c in grp]
        if not self.is_fitted:
            order_idx = sorted(
                range(m),
                key=lambda k: (
                    reps[k].get("panel_idx", 0),
                    (reps[k]["bbox"][1] + reps[k]["bbox"][3] / 2.0) / H - 0.35 * ((reps[k]["bbox"][0] + reps[k]["bbox"][2] / 2.0) / W),
                ),
            )
            return [c for k in order_idx for c in grouped[k]]

        pair_feats = []
        pair_indices = []
        for i in range(m):
            for j in range(m):
                if i != j:
                    pair_feats.append(pairwise_order_features(reps[i], reps[j], W, H))
                    pair_indices.append((i, j))
        Xp = self.scaler.transform(np.vstack(pair_feats))
        probs = 0.65 * self.gbdt.predict_proba(Xp)[:, 1] + 0.35 * self.logreg.predict_proba(Xp)[:, 1]

        P = np.full((m, m), 0.5, dtype=np.float64)
        for idx, (i, j) in enumerate(pair_indices):
            ci, cj = reps[i], reps[j]
            pi, pj = ci.get("panel_idx", 0), cj.get("panel_idx", 0)
            si = (ci["bbox"][1] + ci["bbox"][3] / 2.0) / H - 0.35 * ((ci["bbox"][0] + ci["bbox"][2] / 2.0) / W)
            sj = (cj["bbox"][1] + cj["bbox"][3] / 2.0) / H - 0.35 * ((cj["bbox"][0] + cj["bbox"][2] / 2.0) / W)
            if pi != pj:
                struct_p = 0.92 if pi < pj else 0.08
            else:
                struct_p = 1.0 / (1.0 + math.exp(-12.0 * (sj - si)))
            P[i, j] = 0.60 * probs[idx] + 0.40 * struct_p
        P_sym = 0.5 * (P + (1.0 - P.T))
        np.fill_diagonal(P_sym, 0.5)
        log_W = np.log(np.clip(P_sym, 1e-4, 1.0 - 1e-4))

        if m <= 13:
            N_masks = 1 << m
            dp = np.full(N_masks, -1e18, dtype=np.float64)
            parent_choice = np.full(N_masks, -1, dtype=np.int32)
            dp[0] = 0.0
            for mask in range(N_masks - 1):
                if dp[mask] <= -1e17:
                    continue
                for nxt in range(m):
                    if not (mask & (1 << nxt)):
                        gain = 0.0
                        sub = mask
                        while sub:
                            lsb = sub & -sub
                            prev_idx = lsb.bit_length() - 1
                            gain += log_W[prev_idx, nxt]
                            sub ^= lsb
                        new_mask = mask | (1 << nxt)
                        cand_val = dp[mask] + gain
                        if cand_val > dp[new_mask]:
                            dp[new_mask] = cand_val
                            parent_choice[new_mask] = nxt
            order = []
            cur = N_masks - 1
            while cur > 0:
                last = int(parent_choice[cur])
                order.append(last)
                cur ^= (1 << last)
            order.reverse()
            return [c for i in order for c in grouped[i]]
        else:
            scores = log_W.sum(axis=1) - log_W.sum(axis=0)
            order = list(np.argsort(-scores))
            improved = True
            while improved:
                improved = False
                for k in range(m - 1):
                    a, b = order[k], order[k + 1]
                    if log_W[b, a] > log_W[a, b] + 1e-6:
                        order[k], order[k + 1] = b, a
                        improved = True
            return [c for i in order for c in grouped[i]]


class NarrationClassifier:
    """Distinguishes NARRATION boxes from character dialogue/thought bubbles."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.scaler = StandardScaler()
        self.word_tfidf = TfidfVectorizer(
            ngram_range=(1, 2), max_features=250, sublinear_tf=True
        )
        self.char_tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), max_features=200, sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=20, random_state=seed)
        self.gbdt = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, random_state=seed
        )
        self.logreg = LogisticRegression(C=0.8, class_weight="balanced", random_state=seed)
        self.threshold = 0.58
        self.is_fitted = False

    def _extract_narration_features(self, candidates: List[Dict[str, Any]], fit: bool = False) -> np.ndarray:
        dense_list = []
        texts = []
        for c in candidates:
            txt = c.get("text", "")
            texts.append(txt)
            f = c["features"]
            has_1p = 1.0 if re.search(r"\b(i|i'm|i'll|i'd|i've|my|me|mine|we|we're|our|us)\b", txt, re.I) else 0.0
            has_2p = 1.0 if re.search(r"\b(you|you're|you'll|your|yours)\b", txt, re.I) else 0.0
            has_3p = 1.0 if re.search(r"\b(he|she|his|her|him|they|their|them|himself|herself)\b", txt, re.I) else 0.0
            has_narr_words = 1.0 if re.search(
                r"\b(academy|meanwhile|however|thus|later|next day|finally|arrived|began|thought|realizes|"
                r"realized|dilemma|victory|victor|techniques|passed down|family|zaibatsu|centuries|"
                r"march|april|may|june|july|august|september|october|november|december|19\d\d|20\d\d)\b",
                txt,
                re.I,
            ) else 0.0
            has_interj = 1.0 if re.search(
                r"\b(huh|eh|ah|oh|um|uh|er|hey|wait|whaa+t|noo+|gha+|gaa+h|eek|woah|whoa|c'mon|yeah|okay|sorry|thanks)\b",
                txt,
                re.I,
            ) else 0.0
            has_q = 1.0 if "?" in txt else 0.0
            extra = np.array([has_1p, has_2p, has_3p, has_narr_words, has_interj, has_q], dtype=np.float32)
            dense_list.append(np.concatenate([f, extra]))

        dense = np.vstack(dense_list)
        if fit:
            w_mat = self.word_tfidf.fit_transform(texts)
            c_mat = self.char_tfidf.fit_transform(texts)
            from scipy.sparse import hstack
            comb = hstack([w_mat, c_mat])
            n_comp = min(20, max(1, comb.shape[1] - 1), max(1, comb.shape[0] - 1))
            self.svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
            svd_f = self.svd.fit_transform(comb)
            X = np.nan_to_num(np.hstack([dense, svd_f]), nan=0.0)
            return self.scaler.fit_transform(X)
        else:
            w_mat = self.word_tfidf.transform(texts)
            c_mat = self.char_tfidf.transform(texts)
            from scipy.sparse import hstack
            comb = hstack([w_mat, c_mat])
            svd_f = self.svd.transform(comb)
            X = np.nan_to_num(np.hstack([dense, svd_f]), nan=0.0)
            return self.scaler.transform(X)

    def fit(self, candidates: List[Dict[str, Any]], y_narr: np.ndarray) -> None:
        if len(candidates) < 20 or y_narr.sum() < 3:
            return
        X = self._extract_narration_features(candidates, fit=True)
        self.gbdt.fit(X, y_narr)
        self.logreg.fit(X, y_narr)
        self.is_fitted = True

    def predict_proba(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        if not candidates:
            return np.zeros(0, dtype=np.float32)
        if not self.is_fitted:
            return np.zeros(len(candidates), dtype=np.float32)
        X = self._extract_narration_features(candidates, fit=False)
        p = 0.60 * self.gbdt.predict_proba(X)[:, 1] + 0.40 * self.logreg.predict_proba(X)[:, 1]
        for i, c in enumerate(candidates):
            txt = c.get("text", "")
            if "?" in txt and re.search(r"\b(you|me|my|i'm|what's|aren't|don't)\b", txt, re.I):
                p[i] *= 0.25
        return p.astype(np.float32)


VOCATIVE_NAMES = re.compile(
    r"\b(komaba|okawa|yoda|hachiken|mikage|shinomiya|shirogane|president|fujiwara|chika|"
    r"denji|pochita|makima|aki|power|itadori|fushiguro|gojo|sukuna|kayo|jerry|ma'am|sir|"
    r"senpai|sensei|doctor|grandpa|mom|dad|[a-z]+-san|[a-z]+-kun|[a-z]+-chan|[a-z]+-senpai)\b",
    re.IGNORECASE,
)


def extract_vocatives(text: str) -> set:
    return {m.group(1).lower() for m in VOCATIVE_NAMES.finditer(text)}


def pairwise_speaker_features(
    ui: Dict[str, Any],
    uj: Dict[str, Any],
    idx_i: int,
    idx_j: int,
    page_i: int,
    page_j: int,
    within_page_idx_i: int,
    within_page_idx_j: int,
) -> np.ndarray:
    """
    Compute 28 relational features between dialogue turn ui and dialogue turn uj (idx_i < idx_j)
    across a 3-page manga sequence to predict P(same_speaker(ui, uj)).
    """
    same_page = 1.0 if page_i == page_j else 0.0
    page_dist = float(abs(page_j - page_i))
    same_panel = 1.0 if (same_page and ui.get("panel_idx") == uj.get("panel_idx")) else 0.0
    panel_dist = float(abs(uj.get("panel_idx", 0) - ui.get("panel_idx", 0))) if same_page else 3.0
    same_parent = 1.0 if (same_page and ui.get("parent_id") == uj.get("parent_id") and ui.get("source") == "enclosed") else 0.0

    turn_dist_seq = float(idx_j - idx_i)
    turn_dist_page = float(within_page_idx_j - within_page_idx_i) if same_page else 5.0
    is_adj_turn = 1.0 if (same_page and turn_dist_page == 1.0) else 0.0
    is_alt_turn = 1.0 if (same_page and turn_dist_page == 2.0) else 0.0
    is_step3_turn = 1.0 if (same_page and turn_dist_page == 3.0) else 0.0

    fi, fj = ui["features"], uj["features"]
    cxi, cyi, wi, hi = float(fi[0]), float(fi[1]), float(fi[2]), float(fi[3])
    cxj, cyj, wj, hj = float(fj[0]), float(fj[1]), float(fj[2]), float(fj[3])
    pxi, pyi = float(fi[27]), float(fi[28])
    pxj, pyj = float(fj[27]), float(fj[28])

    dx_page = abs(cxi - cxj)
    dy_page = abs(cyi - cyj)
    euclid_page = math.hypot(dx_page, dy_page) if same_page else 1.5
    dx_panel = abs(pxi - pxj)
    dy_panel = abs(pyi - pyj)

    x_ov = max(0.0, min(cxi + wi / 2, cxj + wj / 2) - max(cxi - wi / 2, cxj - wj / 2)) / max(min(wi, wj), 1e-3) if same_page else 0.0
    y_gap = max(0.0, max(cyi - hi / 2, cyj - hj / 2) - min(cyi + hi / 2, cyj + hj / 2)) if same_page else 1.0

    is_vertical_chain = 1.0 if (same_page and y_gap < 0.045 and (x_ov > 0.25 or dx_page < 0.12)) else 0.0
    is_opposite_side_adj = 1.0 if (same_panel and is_adj_turn and dx_panel > 0.25 and not is_vertical_chain) else 0.0
    is_same_side_alt = 1.0 if (same_page and is_alt_turn and dx_page < 0.20) else 0.0

    vci = ui.get("visual_context")
    vcj = uj.get("visual_context")
    if vci is not None and vcj is not None:
        vc_l1 = float(np.abs(vci - vcj).mean())
        norm_i = float(np.linalg.norm(vci))
        norm_j = float(np.linalg.norm(vcj))
        vc_cos = float(np.dot(vci, vcj) / max(norm_i * norm_j, 1e-6))
    else:
        vc_l1, vc_cos = 0.2, 0.8

    font_h_diff = abs(float(fi[18]) - float(fj[18]))
    both_inverted = 1.0 if (fi[14] > 0.5 and fj[14] > 0.5) else 0.0
    inv_mismatch = 1.0 if (fi[14] > 0.5) != (fj[14] > 0.5) else 0.0

    txt_i, txt_j = ui.get("text", ""), uj.get("text", "")
    voc_i = extract_vocatives(txt_i)
    voc_j = extract_vocatives(txt_j)
    same_vocative = 1.0 if (voc_i and voc_j and len(voc_i & voc_j) > 0) else 0.0
    is_qa_pair = 1.0 if (is_adj_turn and "?" in txt_i and not is_vertical_chain and dx_page > 0.14) else 0.0
    both_first_person = 1.0 if (fi[48] > 0.5 and fj[48] > 0.5) else 0.0

    return np.array([
        same_page, page_dist, same_panel, panel_dist, same_parent,
        1.0 / turn_dist_seq, turn_dist_page / 10.0,
        is_adj_turn, is_alt_turn, is_step3_turn,
        dx_page, dy_page, euclid_page, dx_panel, dy_panel,
        x_ov, y_gap, is_vertical_chain, is_opposite_side_adj, is_same_side_alt,
        vc_l1, vc_cos, font_h_diff, both_inverted, inv_mismatch,
        same_vocative, is_qa_pair, both_first_person,
    ], dtype=np.float32)


class SpeakerAttributionModel:
    """
    Learned pairwise co-reference model + sequence speaker-count estimator + constrained
    agglomerative clustering for consistent speaker labels across 3-page sequences.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.pair_scaler = StandardScaler()
        self.pair_gbdt = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.05, max_leaf_nodes=25, min_samples_leaf=15, random_state=seed
        )
        self.pair_logreg = LogisticRegression(C=1.0, random_state=seed)
        self.count_scaler = StandardScaler()
        self.count_model = Ridge(alpha=2.0, random_state=seed)
        self.is_fitted = False

    def fit(
        self,
        X_pairs: np.ndarray,
        y_pairs: np.ndarray,
        X_seq_counts: np.ndarray,
        y_seq_counts: np.ndarray,
    ) -> None:
        if len(X_pairs) >= 20:
            Xps = self.pair_scaler.fit_transform(X_pairs)
            self.pair_gbdt.fit(Xps, y_pairs)
            self.pair_logreg.fit(Xps, y_pairs)
            self.is_fitted = True
        if len(X_seq_counts) >= 10:
            Xcs = self.count_scaler.fit_transform(X_seq_counts)
            self.count_model.fit(Xcs, y_seq_counts)

    @staticmethod
    def sequence_count_features(
        dialogue_items: List[Tuple[int, int, Dict[str, Any]]],
        mean_same_prob: float,
    ) -> np.ndarray:
        n_turns = len(dialogue_items)
        if n_turns == 0:
            return np.zeros(8, dtype=np.float32)
        active_pages = len({p for p, _, _ in dialogue_items})
        panels = len({(p, c.get("panel_idx", 0)) for p, _, c in dialogue_items})
        cxs = [float(c["features"][0]) for _, _, c in dialogue_items]
        x_std = float(np.std(cxs)) if len(cxs) > 1 else 0.0
        vocs = set()
        q_count = 0
        for _, _, c in dialogue_items:
            t = c.get("text", "")
            vocs |= extract_vocatives(t)
            if "?" in t:
                q_count += 1
        return np.array([
            n_turns / 25.0,
            active_pages / 3.0,
            panels / 12.0,
            x_std,
            len(vocs) / 5.0,
            q_count / float(max(n_turns, 1)),
            mean_same_prob,
            math.sqrt(n_turns) / 5.0,
        ], dtype=np.float32)

    def assign_speakers(
        self,
        pages_candidates: List[List[Dict[str, Any]]],
        narration_classifier: Optional[NarrationClassifier] = None,
        mode: str = "learned",
    ) -> List[List[Dict[str, str]]]:
        merged_pages: List[List[Dict[str, Any]]] = []
        for p_cands in pages_candidates:
            merged: List[Dict[str, Any]] = []
            for c in p_cands:
                if (
                    merged
                    and c.get("source") == "enclosed"
                    and merged[-1].get("source") == "enclosed"
                    and c.get("parent_id") == merged[-1].get("parent_id")
                ):
                    merged[-1] = dict(merged[-1])
                    merged[-1]["text"] = (merged[-1]["text"].rstrip() + " " + c["text"].lstrip()).strip()
                else:
                    merged.append(dict(c))
            merged_pages.append(merged)

        all_cands = [c for p in merged_pages for c in p]
        if not all_cands:
            return [[] for _ in merged_pages]

        if narration_classifier is not None and narration_classifier.is_fitted and mode == "learned":
            narr_probs = narration_classifier.predict_proba(all_cands)
        else:
            narr_probs = np.zeros(len(all_cands), dtype=np.float32)

        dialogue_items: List[Tuple[int, int, Dict[str, Any]]] = []
        is_narr_map: Dict[Tuple[int, int], bool] = {}
        flat_idx = 0
        for p_idx, p_cands in enumerate(merged_pages):
            for c_idx, c in enumerate(p_cands):
                if narr_probs[flat_idx] >= (narration_classifier.threshold if narration_classifier else 0.58):
                    is_narr_map[(p_idx, c_idx)] = True
                else:
                    is_narr_map[(p_idx, c_idx)] = False
                    dialogue_items.append((p_idx, c_idx, c))
                flat_idx += 1

        n_dial = len(dialogue_items)
        dial_labels: List[str] = []

        if n_dial == 0:
            pass
        elif mode == "single_speaker" or n_dial == 1:
            dial_labels = ["character1"] * n_dial
        elif mode == "alternating":
            dial_labels = [f"character{(i % 2) + 1}" for i in range(n_dial)]
        elif mode == "spatial_kmeans" or not self.is_fitted:
            k_clusters = min(3, n_dial)
            feats = np.array([[float(c["features"][0]), float(c["features"][27]), (i % 2) * 0.35] for i, (_, _, c) in enumerate(dialogue_items)])
            Z = linkage(feats, method="ward")
            raw_cl = fcluster(Z, t=k_clusters, criterion="maxclust")
            remap_s: Dict[int, str] = {}
            for cl in raw_cl:
                if int(cl) not in remap_s:
                    remap_s[int(cl)] = f"character{len(remap_s) + 1}"
                dial_labels.append(remap_s[int(cl)])
        else:
            pair_feats = []
            pair_ij = []
            for i in range(n_dial):
                pi, ci_idx, ui = dialogue_items[i]
                for j in range(i + 1, n_dial):
                    pj, cj_idx, uj = dialogue_items[j]
                    pair_feats.append(pairwise_speaker_features(ui, uj, i, j, pi, pj, ci_idx, cj_idx))
                    pair_ij.append((i, j))

            Xp = self.pair_scaler.transform(np.vstack(pair_feats))
            same_probs = 0.65 * self.pair_gbdt.predict_proba(Xp)[:, 1] + 0.35 * self.pair_logreg.predict_proba(Xp)[:, 1]

            D = np.zeros((n_dial, n_dial), dtype=np.float64)
            for idx, (i, j) in enumerate(pair_ij):
                fi = dialogue_items[i][2]["features"]
                fj = dialogue_items[j][2]["features"]
                pi, ci_idx, _ = dialogue_items[i]
                pj, cj_idx, _ = dialogue_items[j]
                p_same = float(same_probs[idx])

                cxi, cyi, wi, hi = float(fi[0]), float(fi[1]), float(fi[2]), float(fi[3])
                cxj, cyj, wj, hj = float(fj[0]), float(fj[1]), float(fj[2]), float(fj[3])
                if pi == pj and abs(cj_idx - ci_idx) == 1:
                    x_ov = max(0.0, min(cxi + wi / 2, cxj + wj / 2) - max(cxi - wi / 2, cxj - wj / 2)) / max(min(wi, wj), 1e-3)
                    y_gap = max(0.0, max(cyi - hi / 2, cyj - hj / 2) - min(cyi + hi / 2, cyj + hj / 2))
                    if y_gap < 0.045 and (x_ov > 0.25 or abs(cxi - cxj) < 0.11):
                        p_same = max(p_same, 0.82)
                    elif abs(cxi - cxj) > 0.24 and y_gap >= 0.01:
                        p_same = min(p_same, 0.22)

                dist_ij = np.clip(1.0 - p_same, 0.02, 0.98)
                D[i, j] = D[j, i] = dist_ij

            seq_f = self.sequence_count_features(dialogue_items, float(np.mean(same_probs)))
            pred_k_raw = float(self.count_model.predict(self.count_scaler.transform(seq_f.reshape(1, -1)))[0])
            if n_dial <= 2:
                k_opt = n_dial
            elif n_dial <= 5:
                k_opt = 2
            else:
                k_opt = int(np.clip(round(pred_k_raw - 0.25), 2, min(4, n_dial)))

            cond_D = squareform(D, checks=False)
            Z = linkage(cond_D, method="average")
            raw_clusters = fcluster(Z, t=k_opt, criterion="maxclust")

            remap: Dict[int, str] = {}
            for cl in raw_clusters:
                if int(cl) not in remap:
                    remap[int(cl)] = f"character{len(remap) + 1}"
                dial_labels.append(remap[int(cl)])

        out_pages: List[List[Dict[str, str]]] = []
        d_ptr = 0
        for p_idx, p_cands in enumerate(merged_pages):
            page_out: List[Dict[str, str]] = []
            for c_idx, c in enumerate(p_cands):
                txt = c["text"].strip()
                if not txt:
                    continue
                if is_narr_map.get((p_idx, c_idx), False):
                    spk = "NARRATION"
                else:
                    spk = dial_labels[d_ptr]
                    d_ptr += 1
                page_out.append({"speaker": spk, "text": txt})
            out_pages.append(page_out)
        return out_pages
