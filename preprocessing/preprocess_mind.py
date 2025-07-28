#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_mind.py
Preprocess MINDsmall_train only, plus optional PLM text embeddings for news.

Outputs (root: --out_dir, default processed/mind_small):
  impr/{train,valid,test}.jsonl      # per-impression ranking (user-level split 80/10/10)
  indices/{user2index.tsv,news2index.tsv}
  text/news.text                     # cleaned concatenated title+abstract
  emb/{dataset_tag}.feat{1|2}{CLS|Mean}   # optional PLM embeddings (binary float32),
                                         # order-aligned with news2index.tsv

Per-impression split requirement:
  - Split USERS (not impressions) into 80%/10%/10% = train/valid/test by a random seed.
  - DO NOT sort impressions chronologically; keep the order in behaviors.tsv.
  - For each impression example, input is 'hist' (clicked news IDs before impression),
    and outputs are 'cands' (candidate news IDs) with 'labels' (0/1).

Sequential leave-one-out is intentionally omitted here to match your last file.
(If you want it back behind a flag, I can add it.)
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional deps for PLM
import torch
from transformers import AutoTokenizer, AutoModel
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


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


def check_path(path: str):
    os.makedirs(path, exist_ok=True)


def set_device(gpu_id: int):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
    else:
        device = torch.device("cpu")
    return device


def load_plm(plm_name: str):
    tok = AutoTokenizer.from_pretrained(plm_name)
    mdl = AutoModel.from_pretrained(plm_name)
    return tok, mdl


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[\r\n]+", " ", s).strip()
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
    news = pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS)
    news = news.drop_duplicates("news_id").set_index("news_id")
    news["text"] = (news["title"].map(clean_text) + " " + news["abstract"].map(clean_text)).str.strip()
    news["title_entities"] = news["title_entities"].map(parse_entities_cell)
    news["abstract_entities"] = news["abstract_entities"].map(parse_entities_cell)
    return news


def load_behaviors(in_dir: str) -> pd.DataFrame:
    path = os.path.join(in_dir, "behaviors.tsv")
    beh = pd.read_csv(path, sep="\t", header=None, names=BEH_COLS)
    beh["hist_list"] = beh["history"].map(parse_history_cell)
    beh["impr_list"] = beh["impressions"].map(parse_impressions_cell)
    # keep a parsed time column for potential sequence construction; impressions split does not use it
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
            fw.write(json.dumps(obj) + "\n")
            n_written += 1
    return n_written


# ---------------------------
# PLM text embedding generation (news)
# ---------------------------
def make_item_text_list(news_df: pd.DataFrame,
                        news2idx: Dict[str, int]) -> List[Tuple[str, str]]:
    """
    Create [(news_id, text)] list for items present in news2idx.
    """
    items = sorted(news2idx.items(), key=lambda x: x[1])  # sort by index
    item_text_list = []
    for nid, _ in items:
        text = news_df.loc[nid, "text"]
        if not isinstance(text, str) or not text:
            text = "."
        item_text_list.append((nid, text))
    return item_text_list


def generate_item_embedding(args,
                            item_text_list: List[Tuple[str, str]],
                            item2index: Dict[str, int],
                            tokenizer,
                            model,
                            device,
                            word_drop_ratio: float = -1.0):
    """
    Generates embeddings (CLS or Mean) for items, order-aligned to item2index.
    Saves to: {out_dir}/emb/{dataset_tag}.feat{1|2}{emb_type}
    """
    print(f"Generate Text Embedding by {args.emb_type}: ")
    print("  PLM:", args.plm_name)
    print("  items:", len(item_text_list))

    # build ordered texts aligned with indices [0..N-1]
    order_texts = [""] * len(item2index)
    for item, text in item_text_list:
        idx = item2index.get(item, None)
        if idx is not None:
            order_texts[idx] = text
    for i, t in enumerate(order_texts):
        if t == "":
            order_texts[i] = "."

    model = model.to(device)
    model.eval()

    batch_size = args.plm_batch_size
    embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(order_texts), batch_size), desc="Embedding"):
            texts = order_texts[start:start + batch_size]

            # optional word drop
            if word_drop_ratio is not None and word_drop_ratio > 0:
                dropped = []
                for sent in texts:
                    toks = sent.split()
                    keep = [w for w in toks if np.random.rand() > word_drop_ratio]
                    s2 = " ".join(keep) if keep else "."
                    dropped.append(s2)
                texts = dropped

            enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state  # [B, L, H]

            if args.emb_type == "CLS":
                vec = out[:, 0, :].detach().cpu()  # [B, H]
            elif args.emb_type == "Mean":
                # mean over non-CLS tokens
                attn = enc["attention_mask"].unsqueeze(-1)  # [B, L, 1]
                masked = out * attn
                denom = (attn[:, 1:, :].sum(dim=1).clamp(min=1)).float()  # [B,1]
                vec = masked[:, 1:, :].sum(dim=1) / denom  # [B, H]
                vec = vec.detach().cpu()
            else:
                raise ValueError(f"Unknown emb_type: {args.emb_type}")

            embs.append(vec)

    embs = torch.cat(embs, dim=0).numpy().astype("float32")
    print("  Embeddings shape:", embs.shape)

    # suffix per Amazon convention: 1 = no word drop, 2 = with word drop
    suffix = "2" if (word_drop_ratio is not None and word_drop_ratio > 0) else "1"

    emb_dir = os.path.join(args.out_dir, "emb")
    check_path(emb_dir)
    out_file = os.path.join(emb_dir, f"{args.dataset_tag}.feat{suffix}{args.emb_type}")
    embs.tofile(out_file)
    print("  Saved to:", out_file)


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
            f.write(f"{nid}\t{idx}\n")

    with open(os.path.join(out_root, "indices", "user2index.tsv"), "w") as f:
        for uid, idx in sorted(user2idx.items(), key=lambda x: x[1]):
            f.write(f"{uid}\t{idx}\n")

    with open(os.path.join(out_root, "text", "news.text"), "w") as f:
        f.write("news_id\ttext\n")
        inv = {v: k for k, v in news2idx.items()}
        for i in range(len(inv)):
            nid = inv[i]
            f.write(f"{nid}\t{news.loc[nid, 'text']}\n")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Preprocess MINDsmall_train into impression format and (optional) PLM embeddings.")
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

    # PLM options
    ap.add_argument("--do_plm", action="store_true",
                    help="If set, compute PLM embeddings for news text")
    ap.add_argument("--plm_name", type=str, default="bert-base-uncased",
                    help="HF model name for the PLM")
    ap.add_argument("--emb_type", type=str, default="CLS", choices=["CLS","Mean"],
                    help="Pooling type for item text embedding")
    ap.add_argument("--word_drop_ratio", type=float, default=-1.0,
                    help="Word drop ratio; if >0, also saves a '.feat2{emb_type}' file")
    ap.add_argument("--gpu_id", type=int, default=0,
                    help="CUDA GPU id to use if available")
    ap.add_argument("--plm_batch_size", type=int, default=32,
                    help="Batch size for PLM encoding")
    ap.add_argument("--dataset_tag", type=str, default="mind_small",
                    help="Tag used when saving embeddings (prefix of .feat files)")

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    print("[1/5] Loading news...")
    news = load_news(args.in_dir)
    print(f"  news rows: {len(news)}")

    print("[2/5] Loading behaviors...")
    beh = load_behaviors(args.in_dir)
    print(f"  impressions: {len(beh)}  users: {beh['user_id'].nunique()}")

    print("[3/5] Building indices...")
    news2idx, user2idx = build_indices(news, beh)
    print(f"  indexed news: {len(news2idx)}  indexed users: {len(user2idx)}")

    print("[4/5] Splitting USERS (80/10/10 by default)...")
    user_ids = sorted(user2idx.keys())
    tr_users, va_users, te_users = split_users(user_ids, seed=args.seed, ratios=tuple(args.user_split))
    print(f"  users -> train={len(tr_users)} valid={len(va_users)} test={len(te_users)}")

    print("[5/5] Writing PER-IMPRESSION jsonl (keep behaviors order)...")
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

    print("Saving indices and text...")
    dump_indices_and_text(news, news2idx, user2idx, args.out_dir)

    if args.do_plm:
        print("Generating PLM embeddings...")
        device = set_device(args.gpu_id)
        tok, mdl = load_plm(args.plm_name)

        item_text_list = make_item_text_list(news, news2idx)
        # .feat1{emb_type}: no word drop
        generate_item_embedding(args, item_text_list, news2idx, tok, mdl, device, word_drop_ratio=-1.0)
        # .feat2{emb_type}: with word drop (if requested)
        if args.word_drop_ratio and args.word_drop_ratio > 0:
            generate_item_embedding(args, item_text_list, news2idx, tok, mdl, device, word_drop_ratio=args.word_drop_ratio)

    print("Done.")


if __name__ == "__main__":
    main()


"""
python preprocessing/preprocess_mind.py \
  --in_dir mind_data/MINDsmall_train \
  --out_dir processed/mind_small \
  --max_hist_len 50 \
  --seed 42 \
  --user_split 0.8 0.1 0.1
"""