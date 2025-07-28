#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_mind.py
Preprocess MINDsmall_train only.

Outputs:
  processed/mind_small/
    impr/{train,valid,test}.jsonl          # per-impression ranking (user-level split 80/10/10)
    seq/{train,valid,test}.inter           # sequential leave-one-out (unchanged from earlier)
    indices/{user2index.tsv,news2index.tsv}
    text/news.text

Per-impression split requirement (as requested):
  - Split USERS (not impressions) into 80%/10%/10% = train/valid/test by a random seed.
  - DO NOT sort impressions chronologically; keep the order in behaviors.tsv.
  - For each impression example, the input is the user's history (list of clicked news IDs),
    and the outputs are candidate news IDs with labels (0/1) from that impression.

Sequential leave-one-out:
  - Build per-user clicked-news sequences in chronological order and do leave-one-out:
      * train: all prefix->next pairs except the last two clicks
      * valid: predict the second-to-last item from its prefix
      * test : predict the last item from its prefix
"""

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


NEWS_COLS = [
    "news_id", "category", "subcategory", "title", "abstract", "url",
    "title_entities", "abstract_entities"
]
BEH_COLS = ["imp_id", "user_id", "time", "history", "impressions"]
TIME_FMT = "%m/%d/%Y %I:%M:%S %p"   # "MM/DD/YYYY HH:MM:SS AM/PM"


# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[\\r\\n]+", " ", s).strip()
    return s[:2000]  # safety cap


def parse_entities_cell(s: str):
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        return json.loads(s)
    except Exception:
        # some dumps contain single quotes
        return json.loads(s.replace("'", '"'))


def parse_history_cell(x: str) -> List[str]:
    if pd.isna(x) or not x:
        return []
    return x.split()


def parse_impressions_cell(x: str) -> List[Tuple[str, int]]:
    """Return list of (news_id, label) where label in {0,1} or None if unlabeled."""
    out = []
    for tok in x.split():
        if "-" in tok:
            nid, lab = tok.split("-")
            out.append((nid, int(lab)))
        else:
            out.append((tok, None))  # test-style (unlabeled); unlikely in MINDsmall_train
    return out


def parse_time(s: str) -> datetime:
    return datetime.strptime(s, TIME_FMT)


# ---------------------------
# Loaders
# ---------------------------
def load_news(in_dir: str) -> pd.DataFrame:
    path = os.path.join(in_dir, "news.tsv")
    news = pd.read_csv(path, sep="\\t", header=None, names=NEWS_COLS)
    news = news.drop_duplicates("news_id").set_index("news_id")
    news["text"] = (news["title"].map(clean_text) + " " + news["abstract"].map(clean_text)).str.strip()
    news["title_entities"] = news["title_entities"].map(parse_entities_cell)
    news["abstract_entities"] = news["abstract_entities"].map(parse_entities_cell)
    return news


def load_behaviors(in_dir: str) -> pd.DataFrame:
    path = os.path.join(in_dir, "behaviors.tsv")
    beh = pd.read_csv(path, sep="\\t", header=None, names=BEH_COLS)
    beh["hist_list"] = beh["history"].map(parse_history_cell)
    beh["impr_list"] = beh["impressions"].map(parse_impressions_cell)
    # keep a parsed time column for sequence construction; impressions split does not use it
    beh["dt"] = beh["time"].map(parse_time)
    return beh


# ---------------------------
# Indexing
# ---------------------------
def build_indices(news: pd.DataFrame, beh: pd.DataFrame):
    used_news = set(news.index.tolist())
    seen_news = set()
    users = set(beh["user_id"].tolist())

    for h in beh["hist_list"]:
        seen_news.update(h)
    for im in beh["impr_list"]:
        for nid, _ in im:
            seen_news.add(nid)

    used_news = used_news & seen_news
    news2idx = {nid: i for i, nid in enumerate(sorted(used_news))}
    user2idx = {uid: i for i, uid in enumerate(sorted(users))}
    return news2idx, user2idx


# ---------------------------
# User-level split for impressions
# ---------------------------
def split_users(user_ids: List[str], seed: int, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-8, "ratios must sum to 1"
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(user_ids))
    n = len(user_ids)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    idx_tr = order[:n_tr]
    idx_va = order[n_tr:n_tr + n_va]
    idx_te = order[n_tr + n_va:]
    tr_users = set(user_ids[i] for i in idx_tr)
    va_users = set(user_ids[i] for i in idx_va)
    te_users = set(user_ids[i] for i in idx_te)
    return tr_users, va_users, te_users


# ---------------------------
# Dumpers (per-impression ranking)
# ---------------------------
def write_impression_jsonl_for_users(beh: pd.DataFrame,
                                     news2idx: Dict[str, int],
                                     user2idx: Dict[str, int],
                                     target_users: set,
                                     out_path: str,
                                     max_hist_len: int = 50) -> int:
    """Write impressions for the selected users. Keep row order as in behaviors.tsv."""
    ensure_dir(os.path.dirname(out_path))
    n_written = 0
    with open(out_path, "w") as fw:
        for _, r in beh.iterrows():
            uid = r["user_id"]
            if uid not in target_users:
                continue

            hist = [news2idx[n] for n in r["hist_list"] if n in news2idx]
            hist = hist[-max_hist_len:]

            cands, labels = [], []
            for nid, lab in r["impr_list"]:
                if nid in news2idx and lab is not None:
                    cands.append(news2idx[nid])
                    labels.append(int(lab))

            if not cands:
                continue

            obj = dict(
                imp_id=str(r["imp_id"]),
                user_id=int(user2idx[uid]),
                time=str(r["time"]),   # kept for reference; not used in splitting
                hist=hist,
                cands=cands,
                labels=labels,
            )
            fw.write(json.dumps(obj) + "\\n")
            n_written += 1
    return n_written


# ---------------------------
# Sequential leave-one-out (unchanged)
# ---------------------------
def build_click_sequences(beh: pd.DataFrame,
                          news2idx: Dict[str, int],
                          user2idx: Dict[str, int]) -> Dict[int, List[int]]:
    """
    Returns dict uidx -> sorted list of clicked news indices (positives only),
    sorted chronologically by dt, then imp_id to stabilize ties.
    """
    clicks = defaultdict(list)
    beh_sorted = beh.sort_values(["user_id", "dt", "imp_id"])
    for _, r in beh_sorted.iterrows():
        u = user2idx.get(r["user_id"])
        if u is None:
            continue
        for nid, lab in r["impr_list"]:
            if lab == 1 and (nid in news2idx):
                clicks[u].append((r["dt"], r["imp_id"], news2idx[nid]))
    seq = {}
    for u, lst in clicks.items():
        lst.sort(key=lambda x: (x[0], x[1]))
        seq[u] = [i for _, __, i in lst]
    return seq


def write_seq_inter(u2seq: Dict[int, List[int]], out_dir: str, max_hist_len: int = 50):
    ensure_dir(out_dir)
    train_path = os.path.join(out_dir, "train.inter")
    valid_path = os.path.join(out_dir, "valid.inter")
    test_path = os.path.join(out_dir, "test.inter")

    with open(train_path, "w") as ftr, \
         open(valid_path, "w") as fva, \
         open(test_path, "w") as fte:
        hdr = "user_id:token\\titem_id_list:token_seq\\titem_id:token\\n"
        ftr.write(hdr); fva.write(hdr); fte.write(hdr)

        ntr = nva = nte = 0
        for u, seq in u2seq.items():
            if len(seq) < 3:
                # still write train prefixes if any
                for t in range(1, len(seq)):
                    hist = seq[:t][-max_hist_len:]
                    tgt = seq[t]
                    ftr.write(f"{u}\\t{' '.join(map(str, hist))}\\t{tgt}\\n")
                    ntr += 1
                continue

            # train: all prefixes before last two
            for t in range(1, len(seq) - 2 + 1):
                hist = seq[:t][-max_hist_len:]
                tgt = seq[t]
                ftr.write(f"{u}\\t{' '.join(map(str, hist))}\\t{tgt}\\n")
                ntr += 1

            # valid: predict second-to-last
            fva.write(f"{u}\\t{' '.join(map(str, seq[:-2][-max_hist_len:]))}\\t{seq[-2]}\\n"); nva += 1
            # test : predict last
            fte.write(f"{u}\\t{' '.join(map(str, seq[:-1][-max_hist_len:]))}\\t{seq[-1]}\\n");  nte += 1

    return {"train": ntr, "valid": nva, "test": nte}


# ---------------------------
# Save indices and text
# ---------------------------
def dump_indices_and_text(news: pd.DataFrame,
                          news2idx: Dict[str, int],
                          user2idx: Dict[str, int],
                          out_root: str):
    ensure_dir(os.path.join(out_root, "indices"))
    ensure_dir(os.path.join(out_root, "text"))

    with open(os.path.join(out_root, "indices", "news2index.tsv"), "w") as f:
        for nid, idx in sorted(news2idx.items(), key=lambda x: x[1]):
            f.write(f"{nid}\\t{idx}\\n")

    with open(os.path.join(out_root, "indices", "user2index.tsv"), "w") as f:
        for uid, idx in sorted(user2idx.items(), key=lambda x: x[1]):
            f.write(f"{uid}\\t{idx}\\n")

    with open(os.path.join(out_root, "text", "news.text"), "w") as f:
        f.write("news_id\\ttext\\n")
        inv = {v: k for k, v in news2idx.items()}
        for i in range(len(inv)):
            nid = inv[i]
            f.write(f"{nid}\\t{news.loc[nid, 'text']}\\n")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Preprocess MINDsmall_train into impression and sequence formats.")
    ap.add_argument("--in_dir", type=str, default="MINDsmall_train",
                    help="Directory containing behaviors.tsv and news.tsv")
    ap.add_argument("--out_dir", type=str, default="processed/mind_small",
                    help="Output root directory")
    ap.add_argument("--max_hist_len", type=int, default=50,
                    help="Max history length to keep")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for user split")
    ap.add_argument("--user_split", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                    help="Train/valid/test ratios over users (must sum to 1)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    print("[1/6] Loading news...")
    news = load_news(args.in_dir)
    print(f"  news rows: {len(news)}")

    print("[2/6] Loading behaviors...")
    beh = load_behaviors(args.in_dir)
    print(f"  impressions: {len(beh)}  users: {beh['user_id'].nunique()}")

    print("[3/6] Building indices...")
    news2idx, user2idx = build_indices(news, beh)
    print(f"  indexed news: {len(news2idx)}  indexed users: {len(user2idx)}")

    print("[4/6] Splitting USERS (80/10/10 by default)...")
    user_ids = sorted(user2idx.keys())
    tr_users, va_users, te_users = split_users(user_ids, seed=args.seed, ratios=tuple(args.user_split))
    print(f"  users -> train={len(tr_users)} valid={len(va_users)} test={len(te_users)}")

    print("[5/6] Writing PER-IMPRESSION jsonl (keep behaviors order)...")
    impr_dir = os.path.join(args.out_dir, "impr")
    ntr = write_impression_jsonl_for_users(
        beh, news2idx, user2idx, tr_users, os.path.join(impr_dir, "train.jsonl"),
        max_hist_len=args.max_hist_len
    )
    nva = write_impression_jsonl_for_users(
        beh, news2idx, user2idx, va_users, os.path.join(impr_dir, "valid.jsonl"),
        max_hist_len=args.max_hist_len
    )
    nte = write_impression_jsonl_for_users(
        beh, news2idx, user2idx, te_users, os.path.join(impr_dir, "test.jsonl"),
        max_hist_len=args.max_hist_len
    )
    print(f"  written impressions: train={ntr}, valid={nva}, test={nte}")

    print("[6/6] Building SEQUENTIAL leave-one-out splits (unchanged)...")
    seq = build_click_sequences(beh, news2idx, user2idx)
    seq_stats = write_seq_inter(seq, os.path.join(args.out_dir, "seq"),
                                max_hist_len=args.max_hist_len)
    print(f"  written sequences: {seq_stats}")

    print("Saving indices and text...")
    dump_indices_and_text(news, news2idx, user2idx, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
