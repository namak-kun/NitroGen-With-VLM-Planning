"""Probe: do the cached frame-conditioned plan-encoder hiddens carry STRONG directional signal?

Settles the teacher/distillation question for the 2B alignment. For each uuid we mean-pool its
(frame-contextualized, text_only) hidden tokens -> one vector, label it by the chunk's dominant
direction (from the stage2 index). We then measure how separable left vs right (and the full
4-way) are with a simple linear probe (logistic regression, stratified CV) + class-mean cosine.

High separability => the encoder already exposes direction the resampler can extract -> a teacher
(privileged P+ distillation) is likely UNNECESSARY. Low separability => keep the teacher.

Run: PYTHONPATH=. .venv/bin/python planner_poc/probe_hidden_separation.py /tmp/stage2_mm_hidden_2b.pt
"""
import sys
import numpy as np
import torch

HID = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stage2_mm_hidden_2b.pt"
INDEX = sys.argv[2] if len(sys.argv) > 2 else "/tmp/stage2_index_a4.pt"


def mean_pool(e):
    h = e["h"].float()                      # (L, d)
    m = e.get("mask")
    if m is not None:
        keep = ~m.bool() if m.dtype == torch.bool else (m == 0)
        if keep.any():
            h = h[keep]
    return h.mean(0).numpy()


def main():
    hid = torch.load(HID, map_location="cpu")
    idx = torch.load(INDEX, map_location="cpu", weights_only=False)
    X, y = [], []
    for u, e in hid.items():
        if u not in idx:
            continue
        d = idx[u]["dir"]
        X.append(mean_pool(e)); y.append(d)
    X = np.stack(X); y = np.array(y)
    print(f"{len(y)} labeled hiddens, dim={X.shape[1]}")
    from collections import Counter
    print("dir dist:", dict(Counter(y)))

    # L2-normalize for cosine + scale-free probe
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    def class_mean_cos(a, b):
        ma = Xn[y == a].mean(0); mb = Xn[y == b].mean(0)
        ma /= np.linalg.norm(ma) + 1e-8; mb /= np.linalg.norm(mb) + 1e-8
        return float(ma @ mb)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def probe(classes):
        mask = np.isin(y, classes)
        if mask.sum() < 20:
            return None
        Xs, ys = Xn[mask], y[mask]
        # guard: need >=2 of each class
        if min(Counter(ys).values()) < 5:
            return None
        clf = LogisticRegression(max_iter=2000, C=1.0)
        sc = cross_val_score(clf, Xs, ys, cv=5)
        return sc.mean(), sc.std(), int(mask.sum())

    print("\n=== linear-probe accuracy (5-fold CV) ===")
    for pair in [["left", "right"], ["up", "down"]]:
        r = probe(pair)
        if r:
            base = 0.5
            print(f"  {pair[0]:>5} vs {pair[1]:<5}: acc={r[0]:.3f}±{r[1]:.3f} (n={r[2]}, chance={base:.2f}) "
                  f"| class-mean cos={class_mean_cos(pair[0], pair[1]):+.3f}")
    r = probe(["left", "right", "up", "down"])
    if r:
        print(f"  4-way dir   : acc={r[0]:.3f}±{r[1]:.3f} (n={r[2]}, chance=0.25)")

    print("\nINTERPRETATION: left/right acc >> 0.5 (e.g. >0.8) and class-mean cos low (<0.9) => strong\n"
          "directional signal in the encoder; teacher/distill likely unnecessary for the 2B run.")


if __name__ == "__main__":
    main()
