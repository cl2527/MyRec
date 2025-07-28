#!/usr/bin/env python
"""Pre‑processing script for **MINDsmall** when you ONLY have the
`MINDsmall_train/` folder (no official dev/test splits).

It builds three kinds of artefacts under an output directory:

1.  *Per‑impression* ranking JSONL files (train/val/test)
2.  *Sequential* leave‑one‑out `.inter` files (train/val/test)
3.  Index / auxiliary files (news2index.tsv, user2index.tsv, text, optional
    entity‑mean features, optional PLM CLS embeddings)

### Splitting strategy
Because the official `MINDsmall_dev` is missing, we split the **training**
impressions chronologically **per user**:

```
first  80 %  → train
next   10 %  → valid
latest 10 %  → test
```

If a user has < 10 impressions, we fall back to a *global* time‑based split.

### Usage
```bash
python preprocess_mind_small.py \
    --in_dir /path/to/MINDsmall_train \
    --out_dir processed/mindsmall  \
    --max_hist 50 \
    --with_plm  --plm_model distilbert-base-uncased  # optional
```
"""
import argparse
import json
import os
import re
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = AutoModel = None  # PLM not required unless --with_plm


NEWS_COLS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEH_COLS = [
    "imp_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

DT_FMT = "%m/%d/%Y %I:%M:%S %p"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clean_text(s: str) -> str:
    """Basic cleaning for title / abstract."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[\r\n]+", " ", s).strip()
    return s[:2000]


def _parse_entities(raw: str):
    """Parse JSON list in title_entities/abstract_entities columns."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(raw.replace("'", '"'))


def parse_news(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS)
    df = df.drop_duplicates("news_id").set_index("news_id")
    df["text"] = (df["title"].map(_clean_text) + " " + df["abstract"].map(_clean_text)).str.strip()
    df["title_entities"] = df["title_entities"].map(_parse_entities)
    df["abstract_entities"] = df["abstract_entities"].map(_parse_entities)
    return df


def parse_behaviors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=BEH_COLS)

    def p_hist(x):
        return [] if pd.isna(x) or not x else x.split()

    def p_impr(x):
        out = []
        for tok in x.split():
            if "-" in tok:
                nid, lab = tok.split("-")
                out.append((nid, int(lab)))
            else:  # should not happen in train but keep for robustness
                out.append((tok, None))
        return out

    df["hist_list"] = df["history"].map(p_hist)
    df["impr_list"] = df["impressions"].map(p_impr)
    df["dt"] = pd.to_datetime(df["time"], format=DT_FMT)
    return df


# ---------------------------------------------------------------------------
# Main processing functions
# ---------------------------------------------------------------------------

def build_indices(news_ids: List[str], user_ids: List[str]):
    news2idx = {nid: i for i, nid in enumerate(sorted(set(news_ids)))}
    user2idx = {uid: i for i, uid in enumerate(sorted(set(user_ids)))}
    return news2idx, user2idx


def split_behaviors_user_time(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.1):
    """Return three DataFrames (train, val, test) using per‑user chronological split."""
    train_rows, val_rows, test_rows = [], [], []

    for uid, g in df.groupby("user_id"):
        g_sorted = g.sort_values("dt")
        n = len(g_sorted)
        if n < 10:  # fallback: global time split will handle later
            train_rows.append(g_sorted)
            continue
        t_idx = int(n * (1 - test_ratio))
        v_idx = int(n * (1 - test_ratio - val_ratio))
        train_rows.append(g_sorted.iloc[:v_idx])
        val_rows.append(g_sorted.iloc[v_idx:t_idx])
        test_rows.append(g_sorted.iloc[t_idx:])

    tr = pd.concat(train_rows, ignore_index=True)
    va = pd.concat(val_rows, ignore_index=True)
    te = pd.concat(test_rows, ignore_index=True)

    # If some users had <10 impressions, they are only in `tr`. Now do a global split
    if len(va) == 0 or len(te) == 0:
        tr = df.sort_values("dt")
        n = len(tr)
        t_idx = int(n * (1 - test_ratio))
        v_idx = int(n * (1 - test_ratio - val_ratio))
        va = tr.iloc[v_idx:t_idx]
        te = tr.iloc[t_idx:]
        tr = tr.iloc[:v_idx]
    return tr, va, te


def build_impression_jsonl(df: pd.DataFrame, news2idx: Dict[str, int], user2idx: Dict[str, int], out_path: str, max_hist: int = 50):
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fw:
        for _, row in df.iterrows():
            hist = [news2idx[nid] for nid in row["hist_list"] if nid in news2idx][-max_hist:]
            cands, labels = [], []
            for nid, lab in row["impr_list"]:
                if nid in news2idx and lab is not None:
                    cands.append(news2idx[nid])
                    labels.append(int(lab))
            if not cands:
                continue
            sample = {
                "imp_id": str(row["imp_id"]),
                "user_id": user2idx[row["user_id"]],
                "time": row["time"],
                "hist": hist,
                "cands": cands,
                "labels": labels,
            }
            fw.write(json.dumps(sample) + "\n")


def build_seq_leave_one_out(df: pd.DataFrame, news2idx: Dict[str, int], user2idx: Dict[str, int], out_dir: str, max_hist: int = 50):
    """Write train/val/test .inter files for sequential next‑item prediction."""
    os.makedirs(out_dir, exist_ok=True)

    # Build click list per user ordered by dt
    click_per_user = defaultdict(list)
    for _, row in df.iterrows():
        for nid, lab in row["impr_list"]:
            if lab == 1 and nid in news2idx:
                click_per_user[user2idx[row["user_id"]]].append((row["dt"], news2idx[nid]))
    for u in click_per_user:
        click_per_user[u].sort()

    train_f = open(os.path.join(out_dir, "train.inter"), "w")
    valid_f = open(os.path.join(out_dir, "valid.inter"), "w")
    test_f  = open(os.path.join(out_dir, "test.inter"), "w")

    header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
    for f in (train_f, valid_f, test_f):
        f.write(header)

    for u, clicks in click_per_user.items():
        items = [i for _, i in clicks]
        if len(items) < 3:
            continue
        # leave‑one‑out: last → test, second last → val, others for train seq sliding
        for t in range(1, len(items)-1):  # up to len-2 for train
            hist = items[:t][-max_hist:]
            train_f.write(f"{u}\t{' '.join(map(str, hist))}\t{items[t]}\n")
        # val
        valid_f.write(f"{u}\t{' '.join(map(str, items[:-1][-max_hist:]))}\t{items[-1]}\n")
        # test
        test_f.write(f"{u}\t{' '.join(map(str, items[-1:][-max_hist:]))}\t{items[-1]}\n")

    for f in (train_f, valid_f, test_f):
        f.close()


# ---------------------------------------------------------------------------
# Optional: PLM and entity embeddings
# ---------------------------------------------------------------------------

def build_plm_embeddings(news_df: pd.DataFrame, news2idx: Dict[str, int], model_name: str, out_path: str, max_len: int = 128):
    if AutoTokenizer is None:
        raise ImportError("transformers not installed. Install it or skip --with_plm option.")

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).cuda()
    mdl.eval()

    texts = [""] * len(news2idx)
    for nid, idx in news2idx.items():
        texts[idx] = news_df.loc[nid, "text"]

    vecs = []
    B = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), B), desc="PLM encode"):
            batch_txt = texts[i : i + B]
            encoded = tok(batch_txt, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
            out = mdl(**encoded).last_hidden_state[:, 0]  # CLS
            vecs.append(out.cpu())
    emb = torch.cat(vecs, 0).numpy().astype("float32")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    emb.tofile(out_path)


def build_entity_mean(news_df: pd.DataFrame, news2idx: Dict[str, int], ent_path: str, out_path: str, conf_thr: float = 0.5):
    ent_vec = {}
    with open(ent_path) as f:
        for line in f:
            t = line.strip().split()
            ent_vec[t[0]] = np.asarray(list(map(float, t[1:])), dtype="float32")
    dim = len(next(iter(ent_vec.values())))
    mat = np.zeros((len(news2idx), dim), dtype="float32")
    mask = np.zeros((len(news2idx),), dtype=bool)

    def _pool(ent_list):
        ids = [e["WikidataId"] for e in ent_list if e.get("Confidence", 1.0) >= conf_thr and e.get("WikidataId") in ent_vec]
        return np.mean([ent_vec[q] for q in ids], axis=0) if ids else None

    for nid, idx in news2idx.items():
        v = news_df.loc[nid]
        pooled = _pool(v["title_entities"])  # title + abstract entities combined
        pooled2 = _pool(v["abstract_entities"])
        if pooled is None and pooled2 is not None:
            pooled = pooled2
        elif pooled is not None and pooled2 is not None:
            pooled = (pooled + pooled2) / 2
        if pooled is not None:
            mat[idx] = pooled
            mask[idx] = True

    np.save(out_path.replace(".npy", "_mask.npy"), mask)
    np.save(out_path, mat)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Preprocess MINDsmall_train only")
    p.add_argument("--in_dir", required=True, help="Path to MINDsmall_train directory")
    p.add_argument("--out_dir", required=True, help="Output root directory")
    p.add_argument("--max_hist", type=int, default=50, help="History length cap")
    p.add_argument("--with_plm", action="store_true", help="Encode news text with a PLM and save CLS vectors")
    p.add_argument("--plm_model", default="distilbert-base-uncased", help="HF model name (used if --with_plm)")
    p.add_argument("--with_entity", action="store_true", help="Save mean‑pooled entity embeddings per news")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: News
    news_path = os.path.join(args.in_dir, "news.tsv")
    news_df = parse_news(news_path)

    # Step 2: Behaviors
    beh_path = os.path.join(args.in_dir, "behaviors.tsv")
    beh_df = parse_behaviors(beh_path)

    # Step 3: indices
    news_ids_used = set(news_df.index)
    user_ids_used = set(beh_df["user_id"].unique())
    news2idx, user2idx = build_indices(list(news_ids_used), list(user_ids_used))

    # Save indices
    idx_dir = os.path.join(args.out_dir, "indices")
    os.makedirs(idx_dir, exist_ok=True)
    with open(os.path.join(idx_dir, "news2index.tsv"), "w") as f:
        for nid, idx in news2idx.items():
            f.write(f"{nid}\t{idx}\n")
    with open(os.path.join(idx_dir, "user2index.tsv"), "w") as f:
        for uid, idx in user2idx.items():
            f.write(f"{uid}\t{idx}\n")

    # Step 4: Split behaviors
    tr_beh, va_beh, te_beh = split_behaviors_user_time(beh_df)

    # Step 5: Impression JSONL
    impr_dir = os.path.join(args.out_dir, "impr")
    build_impression_jsonl(tr_beh, news2idx, user2idx, os.path.join(impr_dir, "train.jsonl"), args.max_hist)
    build_impression_jsonl(va_beh, news2idx, user2idx, os.path.join(impr_dir, "valid.jsonl"), args.max_hist)
    build_impression_jsonl(te_beh, news2idx, user2idx, os.path.join(impr_dir, "test.jsonl"), args.max_hist)

    # Step 6: Sequential leave‑one‑out
    seq_dir = os.path.join(args.out_dir, "seq")
    build_seq_leave_one_out(beh_df, news2idx, user2idx, seq_dir, args.max_hist)

    # Step 7: Save cleaned text
    text_dir = os.path.join(args.out_dir, "text")
    os.makedirs(text_dir, exist_ok=True)
    with open(os.path.join(text_dir, "news.text"), "w") as f:
        f.write("news_id\ttext\n")
        for nid in sorted(news2idx.keys()):
            f.write(f"{nid}\t{news_df.loc[nid,'text']}\n")

    # Optional PLM
    if args.with_plm:
        build_plm_embeddings(
            news_df, news2idx, args.plm_model,
            out_path=os.path.join(args.out_dir, "emb", "mind_cls.bin"),
        )

    # Optional entity mean
    if args.with_entity:
        build_entity_mean(
            news_df, news2idx,
            ent_path=os.path.join(args.in_dir, "entity_embedding.vec"),
            out_path=os.path.join(args.out_dir, "kg", "news_entity_mean.npy"),
        )

    print("\nPreprocessing complete. Outputs written to:", args.out_dir)


if __name__ == "__main__":
    main()
