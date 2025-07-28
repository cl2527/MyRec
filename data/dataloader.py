from torch.utils.data import DataLoader
from mind_dataset import MindImpressionDataset, collate_impressions

ds = MindImpressionDataset(
    jsonl_path="processed/mind_small/impr/train.jsonl",
    news2idx_tsv="processed/mind_small/indices/news2index.tsv",
    emb_path="processed/mind_small/emb/mind_small.feat1CLS",
    max_hist_len=50,
    max_negs_per_impr=50,   # optional: cap negatives; remove or set None to use all
)
dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_impressions)

batch = next(iter(dl))
print(batch["hist_emb"].shape, batch["cand_emb"].shape, batch["labels"].shape)
# -> [B, Lmax, d], [B, Mmax, d], [B, Mmax]


import torch.nn.functional as F

def forward_and_loss(model, batch):
    # model should return user vector per row from (hist_emb, hist_mask)
    u = model.encode_user(batch["hist_emb"], batch["hist_mask"])           # [B, d]
    # Flatten candidates and score by dot product
    B, Mmax, d = batch["cand_emb"].shape
    cand = batch["cand_emb"].reshape(B * Mmax, d)
    u_exp = u.repeat_interleave(Mmax, dim=0)                               # [B*Mmax, d]
    scores = (u_exp * cand).sum(-1).reshape(B, Mmax)                       # [B, Mmax]

    # Mask out padding cand slots > true_M
    mask = torch.arange(Mmax, device=scores.device)[None, :] < batch["true_M"][:, None]
    scores = scores.masked_fill(~mask, -1e9)

    # If single positive per impression: targets = argmax(labels)
    # For safety, compute soft labels (labels normalized over true_M)
    labels = batch["labels"].float()
    labels = labels * mask.float()
    denom = labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    soft = labels / denom
    loss = -(soft * F.log_softmax(scores, dim=1)).sum(dim=1).mean()
    return loss


def bce_loss(model, batch):
    u = model.encode_user(batch["hist_emb"], batch["hist_mask"])           # [B, d]
    B, Mmax, d = batch["cand_emb"].shape
    cand = batch["cand_emb"].reshape(B * Mmax, d)
    u_exp = u.repeat_interleave(Mmax, dim=0)
    s = (u_exp * cand).sum(-1).reshape(B, Mmax)
    mask = torch.arange(Mmax, device=s.device)[None, :] < batch["true_M"][:, None]
    labels = batch["labels"].float()
    # mask loss for padded cands
    loss = F.binary_cross_entropy_with_logits(s[mask], labels[mask], reduction="mean")
    return loss
