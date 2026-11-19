#!/usr/bin/env python
"""
Apple leaf disease detection: reproduction of Bansal et al. (Agriculture 2021, 11(7), 617)
plus leak-free splits, imbalance handling, ensemble evaluation, and distillation.

Subcommands
-----------
  folds    build leak-free stratified folds (near-duplicate images are kept in the same fold)
  train    fine-tune one backbone on one fold (optionally with knowledge distillation)
  eval     per-model and ensemble metrics from saved validation predictions
  bench    parameters / size / CPU latency of models

Dataset: Kaggle "Plant Pathology 2020 - FGVC7" (train.csv + images/).
Run `python apple_leaf.py <cmd> -h` for options.
"""
import argparse
import math
import os
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]

# short key -> timm model name
BACKBONES = {
    "densenet121": "densenet121",
    "effb7": "tf_efficientnet_b7",
    "effb7_ns": "tf_efficientnet_b7.ns_jpg_in1k",  # NoisyStudent weights
    "student_b0": "efficientnet_b0",
    "student_mnv3": "mobilenetv3_large_100",
}


# ----------------------------------------------------------------------------- utils
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stem_to_key(stem: str) -> str:
    """'student_b0_kd' -> 'student_b0'"""
    return stem.replace("_kd", "")


def build_model(key: str, pretrained: bool = True) -> nn.Module:
    name = BACKBONES.get(key, key)
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES))
    except Exception as e:  # missing tag / no internet / version mismatch
        avail = timm.list_pretrained("*efficientnet_b7*")
        raise SystemExit(
            f"Could not create '{name}' ({e}).\n"
            f"Available EfficientNet-B7 tags in your timm version: {avail}\n"
            f"Edit BACKBONES at the top of this file to use one of them."
        )


# ----------------------------------------------------------------------------- folds
def _dhash(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    return (img[:, 1:] > img[:, :-1]).flatten().astype(np.uint8)  # 64 bits


def cmd_folds(a):
    data = Path(a.data_dir)
    df = pd.read_csv(data / "train.csv")
    df["label"] = df[CLASSES].values.argmax(1)
    df["path"] = df["image_id"].map(lambda i: str(data / "images" / f"{i}.jpg"))
    print("Class counts:\n", df["label"].map(dict(enumerate(CLASSES))).value_counts(), "\n")

    # near-duplicate detection via difference hash -> union-find groups
    hashes = np.stack([_dhash(Path(p)) for p in tqdm(df["path"], desc="hashing")])
    n = len(df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = 0
    for i in range(n):
        d = (hashes[i + 1:] != hashes[i]).sum(1)
        for j in np.where(d <= a.dup_thresh)[0]:
            pairs += 1
            parent[find(i)] = find(i + 1 + j)
    df["group"] = [find(i) for i in range(n)]
    n_groups = df["group"].nunique()
    print(f"Near-duplicate pairs (hamming <= {a.dup_thresh}): {pairs}; "
          f"{n - n_groups} images share a group with another image.")

    skf = StratifiedGroupKFold(n_splits=a.n_folds, shuffle=True, random_state=a.seed)
    df["fold"] = -1
    for f, (_, va) in enumerate(skf.split(df, df["label"], groups=df["group"])):
        df.loc[va, "fold"] = f
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df[["image_id", "path", "label", "fold", "group"]].to_csv(out / "folds.csv", index=False)
    print("\nPer-fold class counts:")
    print(pd.crosstab(df["fold"], df["label"].map(dict(enumerate(CLASSES)))))
    print(f"\nSaved {out / 'folds.csv'}")


# ----------------------------------------------------------------------------- data
def get_transforms(size: int, train: bool):
    if train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=7)], p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()])


class LeafDataset(Dataset):
    def __init__(self, df, size, train, canny=False):
        self.paths = df["path"].tolist()
        self.labels = df["label"].values
        self.ids = df["image_id"].tolist()
        self.tf = get_transforms(size, train)
        self.canny = canny and train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        if self.canny and random.random() < 0.25:  # Canny-edge augmentation (as in the paper)
            edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 100, 200)
            img = cv2.addWeighted(img, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.3, 0)
        x = self.tf(image=img)["image"]
        return x, int(self.labels[i])


# ----------------------------------------------------------------------------- losses / metrics
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma, self.weight, self.ls = gamma, weight, label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             label_smoothing=self.ls, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_metrics(y, p):
    pred = p.argmax(1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, labels=range(len(CLASSES)), zero_division=0)
    m = {"acc": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro"))}
    try:
        m["auc_ovr"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except ValueError:
        m["auc_ovr"] = float("nan")
    for i, c in enumerate(CLASSES):
        m[f"recall_{c}"], m[f"prec_{c}"], m[f"f1_{c}"] = float(rec[i]), float(prec[i]), float(f1[i])
    return m


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            out.append(F.softmax(model(x).float(), 1).cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- train
def cmd_train(a):
    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.model}{a.tag}"

    df = pd.read_csv(out / "folds.csv")
    tr_df, va_df = df[df.fold != a.fold].reset_index(drop=True), df[df.fold == a.fold].reset_index(drop=True)
    if a.limit:  # smoke test on a tiny subset
        tr_df, va_df = tr_df.sample(a.limit, random_state=0), va_df.sample(max(a.limit // 2, 8), random_state=0)
    print(f"[{stem}] fold {a.fold}: train={len(tr_df)} val={len(va_df)} device={device}")

    tr_ds = LeafDataset(tr_df, a.img_size, True, a.canny_aug)
    va_ds = LeafDataset(va_df, a.img_size, False)
    sampler = None
    if a.balanced_sampler:
        cnt = np.bincount(tr_df["label"], minlength=len(CLASSES)).astype(float)
        w = (1.0 / np.maximum(cnt, 1))[tr_df["label"].values]
        sampler = WeightedRandomSampler(w, num_samples=len(tr_df), replacement=True)
    tr_dl = DataLoader(tr_ds, a.bs, shuffle=sampler is None, sampler=sampler,
                       num_workers=a.workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, a.bs * 2, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = build_model(a.model).to(device)
    teachers = []
    for t in a.teachers or []:
        tm = build_model(stem_to_key(t), pretrained=False)
        tm.load_state_dict(torch.load(out / f"{t}_f{a.fold}.pt", map_location="cpu"))
        teachers.append(tm.to(device).eval())
    if teachers:
        print(f"Distilling from: {a.teachers}")

    weight = None
    if a.loss in ("weighted", "focal_weighted"):
        cnt = np.bincount(tr_df["label"], minlength=len(CLASSES)).astype(float)
        wt = len(tr_df) / (len(CLASSES) * np.maximum(cnt, 1))
        weight = torch.tensor(wt / wt.mean(), dtype=torch.float32, device=device)
    if a.loss.startswith("focal"):
        crit = FocalLoss(a.gamma, weight, a.label_smoothing)
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=a.label_smoothing)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = max(len(tr_dl) // a.accum, 1)
    total, warm = steps_per_epoch * a.epochs, steps_per_epoch  # 1 epoch warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best, bad, history = -1.0, 0, []
    for ep in range(a.epochs):
        model.train()
        run, t0 = 0.0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tqdm(tr_dl, desc=f"epoch {ep + 1}/{a.epochs}", leave=False)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = model(x)
                loss = crit(logits.float(), y)
                if teachers:
                    with torch.no_grad():
                        t_logits = torch.stack([t(x).float() for t in teachers]).mean(0)
                    kd = F.kl_div(F.log_softmax(logits.float() / a.temp, 1),
                                  F.softmax(t_logits / a.temp, 1), reduction="batchmean")
                    loss = a.alpha * (a.temp ** 2) * kd + (1 - a.alpha) * loss
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run += loss.item()

        probs = predict(model, va_dl, device, amp)
        m = compute_metrics(va_df["label"].values, probs)
        m.update(epoch=ep + 1, train_loss=run / len(tr_dl), secs=time.time() - t0)
        history.append(m)
        print(f"ep {ep + 1}: loss={m['train_loss']:.4f} acc={m['acc']:.4f} "
              f"macroF1={m['macro_f1']:.4f} recall(multi)={m['recall_multiple_diseases']:.3f}")

        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            torch.save(model.state_dict(), out / f"{stem}_f{a.fold}.pt")
            np.savez(out / f"{stem}_f{a.fold}_val.npz", ids=np.array(va_df["image_id"]),
                     y=va_df["label"].values, probs=probs)
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping.")
                break
    pd.DataFrame(history).to_csv(out / f"{stem}_f{a.fold}_log.csv", index=False)
    print(f"Best val macro-F1: {best:.4f}  -> {out / (stem + f'_f{a.fold}.pt')}")


# ----------------------------------------------------------------------------- eval
def _load_val(out, stem, fold):
    z = np.load(Path(out) / f"{stem}_f{fold}_val.npz", allow_pickle=True)
    return z["ids"], z["y"], z["probs"]


def cmd_eval(a):
    out = Path(a.out_dir)
    rows, oof = [], {}
    names = list(a.runs) + (["ENSEMBLE(" + "+".join(a.ensemble) + ")"] if a.ensemble else [])
    for name in names:
        stems = a.ensemble if name.startswith("ENSEMBLE") else [name]
        fold_metrics, ys, ps = [], [], []
        for f in a.folds:
            loaded = [_load_val(out, s, f) for s in stems]
            ids0, y = loaded[0][0], loaded[0][1]
            assert all((l[0] == ids0).all() for l in loaded), "val id order mismatch"
            p = np.mean([l[2] for l in loaded], axis=0)
            fold_metrics.append(compute_metrics(y, p))
            ys.append(y)
            ps.append(p)
        mdf = pd.DataFrame(fold_metrics)
        mean, std = mdf.mean(), mdf.std(ddof=0)
        rows.append({"run": name, "folds": len(a.folds), "tag": a.note,
                     **{f"{k}": f"{mean[k]:.4f} +/- {std[k]:.4f}" for k in mdf.columns}})
        oof[name] = (np.concatenate(ys), np.concatenate(ps))

    res = pd.DataFrame(rows)
    show = ["run", "acc", "macro_f1", "auc_ovr", "recall_multiple_diseases"]
    print(res[show].to_string(index=False))
    path = out / "results.csv"
    res.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"\nAppended to {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    for name, (y, p) in oof.items():
        cm = confusion_matrix(y, p.argmax(1), labels=range(len(CLASSES)))
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.title(name, fontsize=8)
        plt.tight_layout()
        safe = name.replace("(", "_").replace(")", "").replace("+", "_")
        plt.savefig(out / f"cm_{safe}.png", dpi=150)
        plt.close()


# ----------------------------------------------------------------------------- bench
def cmd_bench(a):
    rows = []
    torch.set_num_threads(a.threads)
    for stem in a.runs:
        m = build_model(stem_to_key(stem), pretrained=False).eval().cpu()
        params = sum(p.numel() for p in m.parameters())
        x = torch.randn(1, 3, a.img_size, a.img_size)
        with torch.no_grad():
            for _ in range(2):
                m(x)
            t0 = time.time()
            for _ in range(a.iters):
                m(x)
        rows.append({"model": stem, "params_M": round(params / 1e6, 2),
                     "size_MB_fp32": round(params * 4 / 1e6, 1),
                     "cpu_ms_per_img": round((time.time() - t0) / a.iters * 1000, 1)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(Path(a.out_dir) / "bench.csv", index=False)


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data_dir", default="/kaggle/input/plant-pathology-2020-fgvc7")
        sp.add_argument("--out_dir", default="outputs")
        sp.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("folds"); common(s)
    s.add_argument("--n_folds", type=int, default=5)
    s.add_argument("--dup_thresh", type=int, default=4, help="max Hamming distance (of 64 bits) to call near-duplicate")
    s.set_defaults(fn=cmd_folds)

    s = sub.add_parser("train"); common(s)
    s.add_argument("--model", required=True, choices=list(BACKBONES))
    s.add_argument("--fold", type=int, default=0)
    s.add_argument("--tag", default="", help="suffix for output names, e.g. _kd")
    s.add_argument("--img_size", type=int, default=384)
    s.add_argument("--epochs", type=int, default=15)
    s.add_argument("--bs", type=int, default=16)
    s.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--wd", type=float, default=1e-2)
    s.add_argument("--patience", type=int, default=5)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--loss", default="ce", choices=["ce", "weighted", "focal", "focal_weighted"])
    s.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    s.add_argument("--label_smoothing", type=float, default=0.0)
    s.add_argument("--balanced_sampler", action="store_true")
    s.add_argument("--canny_aug", action="store_true")
    s.add_argument("--limit", type=int, default=0, help="use N train images only (smoke test)")
    s.add_argument("--teachers", nargs="*", help="run stems of trained teachers for distillation")
    s.add_argument("--alpha", type=float, default=0.7, help="weight of the KD loss")
    s.add_argument("--temp", type=float, default=2.0, help="KD temperature")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval"); common(s)
    s.add_argument("--runs", nargs="+", default=[], help="run stems to evaluate individually")
    s.add_argument("--ensemble", nargs="*", help="run stems to average into one ensemble")
    s.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    s.add_argument("--note", default="", help="free-text label stored in results.csv")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("bench"); common(s)
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--img_size", type=int, default=384)
    s.add_argument("--iters", type=int, default=10)
    s.add_argument("--threads", type=int, default=4)
    s.set_defaults(fn=cmd_bench)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
