# mind_dataset.py
import os, json, math
import numpy as np
import torch
from torch.utils.data import Dataset

def _read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _load_news2idx(path):
    # tsv: news_id \t index
    m = {}
    with open(path, "r") as f:
        for line in f:
            nid, idx = line.rstrip("\n").split("\t")
            m[nid] = int(idx)
    return m

def _memmap_item_emb(emb_path, n_items, dtype=np.float32):
    # Infer embedding dim from file size.
    item_bytes = os.path.getsize(emb_path)
    d = item_bytes // (np.dtype(dtype).itemsize * n_items)
    assert d * n_items * np.dtype(dtype).itemsize == item_bytes, \
        f"Bad emb file size: cannot infer dim; got {item_bytes} bytes for {n_items} items"
    arr = np.memmap(emb_path, dtype=dtype, mode="r", shape=(n_items, d))
    return arr, d

class MindImpressionDataset(Dataset):
    """
    Each __getitem__ returns one impression:
      - hist_emb: [L, d]   (unpadded; collate will pad)
      - hist_mask: [L]
      - cand_emb: [M, d]
      - labels: [M]
      - cand_ids: [M] (long)
      - user_id: int
      - imp_id: str
    """
    def __init__(
        self,
        jsonl_path,                 # impr/train.jsonl (or valid/test)
        news2idx_tsv,               # indices/news2index.tsv
        emb_path,                   # emb/mind_small.feat1CLS (float32 binary)
        max_hist_len=50,
        max_negs_per_impr=None,     # optional: cap #negatives per impression
    ):
        super().__init__()
        self.samples = list(_read_jsonl(jsonl_path))
        self.news2idx = _load_news2idx(news2idx_tsv)
        self.max_hist_len = max_hist_len
        self.max_negs = max_negs_per_impr

        # Build a reverse map length to know how many items there are
        n_items = len(self.news2idx)
        self.item_emb, self.d = _memmap_item_emb(emb_path, n_items)  # np.memmap (N, d)

    def __len__(self):
        return len(self.samples)

    def _index_item(self, nid):
        # nid is already an index (int) in the JSONL; if you stored indices there.
        # If JSONL stores indices (as in our preprocessing), they’re ints already.
        return int(nid)

    def __getitem__(self, idx):
        row = self.samples[idx]
        # JSONL structure: {"imp_id", "user_id", "time", "hist": [idx...], "cands": [idx...], "labels": [0/1...]}

        # --- history ---
        hist_idx = [self._index_item(n) for n in row["hist"]][-self.max_hist_len:]
        if len(hist_idx) == 0:
            hist_emb = np.zeros((0, self.d), dtype=np.float32)
            hist_mask = np.zeros((0,), dtype=np.int64)
        else:
            hist_emb = self.item_emb[hist_idx]  # (L, d)
            hist_mask = np.ones((len(hist_idx),), dtype=np.int64)

        # --- candidates + labels ---
        cand_idx = [self._index_item(n) for n in row["cands"]]
        labels   = np.array(row["labels"], dtype=np.int64)

        if self.max_negs is not None:
            # Keep all positives + sample up to max_negs negatives
            pos_idx = [i for i, y in enumerate(labels) if y == 1]
            neg_idx = [i for i, y in enumerate(labels) if y == 0]
            if len(neg_idx) > self.max_negs:
                sel_neg = np.random.choice(neg_idx, size=self.max_negs, replace=False)
                keep = np.concatenate([pos_idx, sel_neg], axis=0)
                keep.sort()
                cand_idx = [cand_idx[i] for i in keep]
                labels   = labels[keep]

        cand_emb = self.item_emb[cand_idx]  # (M, d)
        cand_ids = np.array(cand_idx, dtype=np.int64)

        out = dict(
            hist_emb=torch.from_numpy(hist_emb),          # [L, d]
            hist_mask=torch.from_numpy(hist_mask),        # [L]
            cand_emb=torch.from_numpy(cand_emb),          # [M, d]
            labels=torch.from_numpy(labels),              # [M]
            cand_ids=torch.from_numpy(cand_ids),          # [M]
            user_id=int(row["user_id"]),
            imp_id=row["imp_id"],
        )
        return out

def collate_impressions(batch):
    """
    Pads histories and candidate sets in a batch.
    Returns:
      hist_emb  [B, Lmax, d]
      hist_mask [B, Lmax]
      cand_emb  [B, Mmax, d]
      labels    [B, Mmax]
      cand_ids  [B, Mmax]
      user_id   [B]
      imp_id    list[str]
      true_M    [B] (original #cands)
    """
    B = len(batch)
    d = batch[0]["hist_emb"].shape[-1] if batch[0]["hist_emb"].numel() > 0 else batch[0]["cand_emb"].shape[-1]
    Ls = [b["hist_emb"].shape[0] for b in batch]
    Ms = [b["cand_emb"].shape[0] for b in batch]
    Lmax, Mmax = (max(Ls) if Ls else 0), max(Ms)

    # Pad histories
    hist_emb = torch.zeros(B, Lmax, d, dtype=torch.float32)
    hist_mask = torch.zeros(B, Lmax, dtype=torch.int64)
    for i, b in enumerate(batch):
        L = b["hist_emb"].shape[0]
        if L:
            hist_emb[i, :L] = b["hist_emb"]
            hist_mask[i, :L] = b["hist_mask"]

    # Pad candidates
    cand_emb = torch.zeros(B, Mmax, d, dtype=torch.float32)
    labels   = torch.zeros(B, Mmax, dtype=torch.int64)
    cand_ids = torch.zeros(B, Mmax, dtype=torch.int64)
    true_M   = torch.tensor(Ms, dtype=torch.int64)
    for i, b in enumerate(batch):
        M = b["cand_emb"].shape[0]
        cand_emb[i, :M] = b["cand_emb"]
        labels[i, :M]   = b["labels"]
        cand_ids[i, :M] = b["cand_ids"]

    user_id = torch.tensor([b["user_id"] for b in batch], dtype=torch.int64)
    imp_id  = [b["imp_id"] for b in batch]

    return dict(
        hist_emb=hist_emb, hist_mask=hist_mask,
        cand_emb=cand_emb, labels=labels, cand_ids=cand_ids,
        user_id=user_id, imp_id=imp_id, true_M=true_M
    )
