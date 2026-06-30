#!/usr/bin/env python3
"""One-day inverse-dynamics feasibility gate on human demo recordings.

The model is intentionally small: a non-causal window of resized grayscale
frames is stacked as channels and fed to a compact CNN that predicts the 12
binary gamepad buttons for the center frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SMW_CRITICAL = {"B_jump": 0, "Y_run": 1, "LEFT": 6, "RIGHT": 7}


@dataclass
class Demo:
    path: Path
    game: str
    name: str
    observations: np.ndarray
    actions: np.ndarray
    buttons: list[str | None]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_buttons(demo_path: Path, npz_buttons: np.ndarray | None = None) -> list[str | None]:
    meta = demo_path.with_name("meta.json")
    if meta.exists():
        with meta.open() as f:
            return json.load(f)["buttons"]
    if npz_buttons is None:
        return [str(i) for i in range(12)]
    out: list[str | None] = []
    for b in npz_buttons.tolist():
        if b is None:
            out.append(None)
        else:
            out.append(str(b))
    return out


def load_demos(root: Path, game: str) -> list[Demo]:
    paths = sorted((root / game).glob("*/demo.npz"))
    demos: list[Demo] = []
    for p in paths:
        z = np.load(p, mmap_mode="r", allow_pickle=True)
        actions = np.asarray(z["actions"], dtype=np.uint8)
        observations = z["observations"]
        buttons = read_buttons(p, z["buttons"] if "buttons" in z.files else None)
        demos.append(Demo(p, game, p.parent.name, observations, actions, buttons))
    if not demos:
        raise FileNotFoundError(f"no demos found for {game} under {root}")
    return demos


def choose_holdout(demos: list[Demo], requested: str | None) -> Demo:
    if requested:
        for d in demos:
            if d.name == requested or str(d.path).endswith(requested):
                return d
        raise ValueError(f"holdout {requested!r} did not match {[d.name for d in demos]}")
    non_idle = [d for d in demos if d.actions.sum() > 0]
    # A medium/long complete playthrough is more informative than a tiny clip.
    return sorted(non_idle, key=lambda d: len(d.actions))[-2 if len(non_idle) > 1 else -1]


def resize_gray_frames(obs: np.ndarray, size: int, batch: int = 512) -> torch.Tensor:
    """Return uint8 tensor [T, size, size] in CPU RAM."""
    outs: list[torch.Tensor] = []
    for i in range(0, len(obs), batch):
        arr = np.asarray(obs[i : i + batch])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        # BT.601 luma
        g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        g = F.interpolate(g, (size, size), mode="area")
        outs.append((g[:, 0].mul(255).round().clamp(0, 255)).to(torch.uint8).cpu())
    return torch.cat(outs, dim=0)


class IDMDataset(Dataset):
    def __init__(
        self,
        demos: list[Demo],
        resized: dict[str, torch.Tensor],
        window: int,
        stride: int,
        max_samples: int | None,
        motion_filter: float,
    ) -> None:
        self.demos = demos
        self.resized = resized
        self.window = window
        self.radius = window // 2
        self.indices: list[tuple[int, int]] = []
        for di, d in enumerate(demos):
            acts = d.actions
            frames = resized[d.name]
            for t in range(0, len(acts), stride):
                if t - self.radius < 0 or t + self.radius >= len(frames):
                    continue
                if motion_filter > 0:
                    a = frames[t - 1].float() if t > 0 else frames[t].float()
                    b = frames[t + 1].float() if t + 1 < len(frames) else frames[t].float()
                    if (a - b).abs().mean().item() < motion_filter:
                        continue
                self.indices.append((di, t))
        if max_samples and len(self.indices) > max_samples:
            rng = random.Random(0)
            self.indices = rng.sample(self.indices, max_samples)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        di, t = self.indices[idx]
        d = self.demos[di]
        frames = self.resized[d.name][t - self.radius : t + self.radius + 1]
        x = frames.float() / 255.0
        y = torch.from_numpy(np.asarray(d.actions[t], dtype=np.float32))
        return x, y

    def targets(self) -> np.ndarray:
        ys = [self.demos[di].actions[t] for di, t in self.indices]
        return np.asarray(ys, dtype=np.float32)


class TinyIDM(nn.Module):
    def __init__(self, in_ch: int, out_dim: int = 12, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(width * 4, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits.append(model(x).detach().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def f1_stats(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j].astype(bool)
        yp = y_pred[:, j].astype(bool)
        tp = int(np.logical_and(yt, yp).sum())
        fp = int(np.logical_and(~yt, yp).sum())
        fn = int(np.logical_and(yt, ~yp).sum())
        tn = int(np.logical_and(~yt, ~yp).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / max(1, len(yt))
        rows.append({"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "support": int(yt.sum()), "pred_pos": int(yp.sum())})
    return rows


def tune_thresholds(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 37)
    for j in range(labels.shape[1]):
        best = (-1.0, 0.5)
        for th in grid:
            f1 = f1_stats(labels[:, [j]], (probs[:, [j]] >= th).astype(np.uint8))[0]["f1"]
            if f1 > best[0]:
                best = (f1, float(th))
        thresholds[j] = best[1]
    return thresholds


def brier_score(logits: np.ndarray, labels: np.ndarray) -> list[float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    return ((probs - labels) ** 2).mean(axis=0).tolist()


def make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    # Upweight frames with rare positive buttons while retaining no-op frames.
    pos = labels.mean(axis=0)
    inv = 1.0 / np.clip(pos, 0.01, 1.0)
    weights = 1.0 + (labels * inv[None]).sum(axis=1)
    weights = np.clip(weights, 1.0, np.percentile(weights, 99))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demos = load_demos(Path(args.demo_root), args.game)
    holdout = choose_holdout(demos, args.holdout)
    train_demos = [d for d in demos if d.name != holdout.name and len(d.actions) >= args.min_demo_actions and d.actions.sum() > 0]
    if not train_demos:
        raise RuntimeError("no non-idle train demos remain")

    print(f"game={args.game} holdout={holdout.name} train={[d.name for d in train_demos]}", flush=True)
    resized: dict[str, torch.Tensor] = {}
    for d in train_demos + [holdout]:
        print(f"resizing {d.name} frames={len(d.observations)}", flush=True)
        resized[d.name] = resize_gray_frames(d.observations, args.size)

    train_ds = IDMDataset(train_demos, resized, args.window, args.stride, args.max_train_samples, args.motion_filter)
    val_ds = IDMDataset([holdout], resized, args.window, args.val_stride, None, 0.0)
    train_labels = train_ds.targets()
    val_labels = val_ds.targets()
    print(f"train_samples={len(train_ds)} val_samples={len(val_ds)} train_pos={train_labels.mean(0).round(4).tolist()} val_pos={val_labels.mean(0).round(4).tolist()}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = TinyIDM(args.window, 12, args.width).to(device)
    pos = train_labels.mean(axis=0)
    pos_weight = (1.0 - pos) / np.clip(pos, 1e-3, 1.0)
    pos_weight = np.clip(pos_weight, 0.5, args.max_pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train_labels) if args.weighted_sampler else None,
        shuffle=not args.weighted_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    best = {"critical_mean_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(x)
            seen += len(x)
        sched.step()
        train_logits, train_y = predict(model, DataLoader(train_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers), device)
        thresholds = tune_thresholds(train_logits, train_y)
        val_logits, val_y = predict(model, val_loader, device)
        val_pred = ((1 / (1 + np.exp(-val_logits))) >= thresholds[None]).astype(np.uint8)
        rows = f1_stats(val_y, val_pred)
        crit = float(np.mean([rows[i]["f1"] for i in SMW_CRITICAL.values()]))
        print(f"epoch={epoch:03d} loss={total/max(1,seen):.4f} critical_mean_f1={crit:.3f} " + " ".join(f"{k}={rows[i]['f1']:.3f}" for k, i in SMW_CRITICAL.items()), flush=True)
        if crit > best["critical_mean_f1"]:
            best = {
                "epoch": epoch,
                "critical_mean_f1": crit,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "thresholds": thresholds.tolist(),
            }

    model.load_state_dict(best["state_dict"])
    train_logits, train_y = predict(model, DataLoader(train_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers), device)
    thresholds = np.asarray(best["thresholds"], dtype=np.float32)
    val_logits, val_y = predict(model, val_loader, device)
    probs = 1 / (1 + np.exp(-val_logits))
    pred = (probs >= thresholds[None]).astype(np.uint8)
    noop = np.zeros_like(val_y, dtype=np.uint8)
    all_one = np.ones_like(val_y, dtype=np.uint8)
    train_majority = (train_y.mean(axis=0) >= 0.5).astype(np.uint8)
    majority = np.repeat(train_majority[None], len(val_y), axis=0)
    # Oracle validation-majority is not a fair deployable baseline, but it is a
    # useful sanity check for buttons that are held for most of one held-out run.
    val_majority = np.repeat((val_y.mean(axis=0) >= 0.5).astype(np.uint8)[None], len(val_y), axis=0)

    buttons = holdout.buttons
    metrics_rows = f1_stats(val_y, pred)
    noop_rows = f1_stats(val_y, noop)
    all_one_rows = f1_stats(val_y, all_one)
    majority_rows = f1_stats(val_y, majority)
    val_majority_rows = f1_stats(val_y, val_majority)
    brier = brier_score(val_logits, val_y)
    per_button = []
    for i in range(12):
        per_button.append(
            {
                "idx": i,
                "button": buttons[i] if i < len(buttons) else str(i),
                "prevalence": float(val_y[:, i].mean()),
                "threshold": float(thresholds[i]),
                "idm": metrics_rows[i],
                "noop_baseline": noop_rows[i],
                "all_one_baseline": all_one_rows[i],
                "majority_baseline": majority_rows[i],
                "val_oracle_majority_baseline": val_majority_rows[i],
                "brier": float(brier[i]),
            }
        )

    trace_start = find_trace_start(val_y, [0, 1, 6, 7], args.trace_len)
    trace_end = min(len(val_y), trace_start + args.trace_len)
    trace_path = out_dir / f"trace_{args.game}_{holdout.name}.csv"
    with trace_path.open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["t"]
        for name, idx in SMW_CRITICAL.items():
            cols += [f"true_{name}", f"pred_{name}", f"prob_{name}"]
        w.writerow(cols)
        for t in range(trace_start, trace_end):
            row: list[object] = [t]
            for _, idx in SMW_CRITICAL.items():
                row += [int(val_y[t, idx]), int(pred[t, idx]), round(float(probs[t, idx]), 4)]
            w.writerow(row)

    model_path = out_dir / f"idm_{args.game}_{holdout.name}.pt"
    torch.save(
        {
            "model": best["state_dict"],
            "thresholds": thresholds,
            "args": vars(args),
            "buttons": buttons,
            "holdout": holdout.name,
            "train_demos": [d.name for d in train_demos],
            "model_class": "TinyIDM",
        },
        model_path,
    )
    metrics = {
        "game": args.game,
        "model": {"class": "TinyIDM", "window": args.window, "size": args.size, "width": args.width, "input": "stacked grayscale frames"},
        "train_demos": [d.name for d in train_demos],
        "holdout_demo": holdout.name,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_epoch": best["epoch"],
        "critical_mean_f1": best["critical_mean_f1"],
        "critical_buttons": SMW_CRITICAL,
        "pos_weight": pos_weight.tolist(),
        "per_button": per_button,
        "trace_csv": str(trace_path),
        "trace_start": trace_start,
        "trace_end": trace_end,
        "model_path": str(model_path),
    }
    metrics_path = out_dir / f"metrics_{args.game}_{holdout.name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"saved model={model_path} metrics={metrics_path} trace={trace_path}", flush=True)
    return metrics


def find_trace_start(y: np.ndarray, buttons: Iterable[int], length: int) -> int:
    activity = y[:, list(buttons)].sum(axis=1)
    if len(activity) <= length:
        return 0
    kernel = np.ones(length, dtype=np.float32)
    scores = np.convolve(activity, kernel, mode="valid")
    return int(scores.argmax())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-root", default="docs/demos/demos")
    p.add_argument("--game", default="SuperMarioWorld-Snes")
    p.add_argument("--holdout")
    p.add_argument("--out-dir", default="docs/idm_gate")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--size", type=int, default=84)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--val-stride", type=int, default=1)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--min-demo-actions", type=int, default=300)
    p.add_argument("--motion-filter", type=float, default=0.0)
    p.add_argument("--max-pos-weight", type=float, default=25.0)
    p.add_argument("--weighted-sampler", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--trace-len", type=int, default=360)
    args = p.parse_args()
    if args.window % 2 != 1:
        raise ValueError("--window must be odd")
    metrics = train(args)
    crit = metrics["critical_mean_f1"]
    print(f"RESULT critical_mean_f1={crit:.3f}", flush=True)


if __name__ == "__main__":
    main()
