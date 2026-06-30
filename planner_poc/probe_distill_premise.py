"""Premise check for the action-augmented DISTILLATION idea (privileged-info teacher):
  base plan P_i   = "I need to go to hyrule castle"                       (student input)
  augmented P+_i  = P_i + " To do this I take: <action summary of chunk>" (teacher input)
Distillation only helps if the TEACHER representation carries separable per-chunk (=> action)
content that the base plan lacks. So compare mean pairwise cosine of the raw VLM last-hidden
(mean-pooled) for base vs augmented prompts over the Stage-2 chunks. Lower cos for augmented
=> appending the action sequence de-collinearizes the representation => a real teacher signal
to distill the base plan toward. Also report base<->augmented per-chunk cosine (how far the
student must move) and whether augmented separates chunks of DIFFERENT dominant direction.
"""
import glob, json, os, sys
import numpy as np
import torch
import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, chunk_dominant_dir
from action_summary import summarize_chunk

device = "cuda"; H = 18
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=f"{REPO}/ckpts/qwen35-0.8b")); pl.load()
cache = PlanHiddenCache(pl, device)
LOOKUP = json.load(open("/tmp/stage2_plan_lookup.json"))


def pooled(text):
    h, kpm = cache.get(text if text else ".")
    mask = (~kpm).float().unsqueeze(-1)
    return ((h * mask).sum(0) / mask.sum().clamp(min=1)).float()


def mean_pairwise_cos(mat, labels=None, same=None):
    x = torch.nn.functional.normalize(mat, dim=-1); s = x @ x.t(); n = s.shape[0]
    if labels is None:
        return float((s.sum() - n) / (n * (n - 1)))
    L = torch.tensor(labels); M = (L[:, None] == L[None, :])
    M.fill_diagonal_(False)
    if not same:
        M = ~M; M.fill_diagonal_(False)
    return float(s[M].mean()) if M.any() else float("nan")


def main():
    base, aug, dirs, act_only = [], [], [], []
    mds = sorted(glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True))
    for md in mds:
        m_ = json.load(open(md)); uuid = m_["uuid"]
        if uuid not in LOOKUP:
            continue
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 303, H, 2) \
            or assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 3, H, 2)
        if rc is None:
            continue
        dd = chunk_dominant_dir(rc)
        p = LOOKUP[uuid]["plan"]
        summ = summarize_chunk(rc)
        base.append(p)
        aug.append(p + " To do this I take the following actions: " + summ + ".")
        act_only.append(summ)
        dirs.append(dd if dd else "idle")
        if len(base) >= 60:
            break
    print(f"chunks: {len(base)}")
    Bt = torch.stack([pooled(t) for t in base])
    At = torch.stack([pooled(t) for t in aug])
    # action-summary ALONE (no base plan, no template) -> is the action CONTENT separable
    # when not diluted by a shared template? If yes, the fix is "isolate the action tokens".
    AsOnly = torch.stack([pooled(t) for t in act_only])
    lab = [hash(d) % 1000 for d in dirs]
    print("\nraw VLM last-hidden (mean-pool) pairwise cosine  (lower = more separable):")
    print(f"  base plans           : {mean_pairwise_cos(Bt):.4f}")
    print(f"  augmented (templated): {mean_pairwise_cos(At):.4f}   <- naive teacher")
    print(f"  action-summary ALONE : {mean_pairwise_cos(AsOnly):.4f}   <- isolated action content")
    print("\nDOES the teacher separate chunks by DOMINANT DIRECTION? (within vs across dir)")
    print(f"  base  : same-dir {mean_pairwise_cos(Bt, lab, True):.4f}  vs across-dir {mean_pairwise_cos(Bt, lab, False):.4f}")
    print(f"  aug   : same-dir {mean_pairwise_cos(At, lab, True):.4f}  vs across-dir {mean_pairwise_cos(At, lab, False):.4f}")
    print(f"  act   : same-dir {mean_pairwise_cos(AsOnly, lab, True):.4f}  vs across-dir {mean_pairwise_cos(AsOnly, lab, False):.4f}")
    # how far the student (base) must move to reach the teacher (aug), per chunk
    cos_ba = torch.nn.functional.cosine_similarity(Bt, At, dim=-1)
    print(f"\nper-chunk base<->augmented cosine: mean {cos_ba.mean():.3f} (1.0 = appending actions changed nothing)")
    print("\nINTERPRETATION: aug << base in across-dir cosine => the action text injects")
    print("separable, direction-correlated content => distilling base->aug should transfer")
    print("finer-than-direction action grounding into the base-plan tokens (test-time: base only).")


if __name__ == "__main__":
    main()
