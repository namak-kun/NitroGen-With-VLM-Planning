"""Summarize a real NitroGen action chunk as natural-language CONTEXT for the VLM. The VLM
cannot infer the gamepad inputs from pixels (EXP-036/037), so we TELL it what the player
actually did and let it explain the intent in game terms — grounding the plan in real
actions by construction.

The canonical implementation now lives in nitrogen.training.actions.summarize_chunk; this
module re-exports it for the POC scripts.

Action layout (per timestep): buttons[0:21], j_left[21:23] (x,y in [-1,1]; -x=left,
-y=up), j_right[23:25]. Chunk = 18 steps.
"""
from nitrogen.training.actions import summarize_chunk, _stick_dir  # noqa: F401


if __name__ == "__main__":
    import sys, glob, os, json
    from nitrogen.training.actions import load_chunk_actions, assemble_chunk
    dirs = sorted(os.path.dirname(md) for md in glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True))
    for d in dirs[:8]:
        m = json.load(open(os.path.join(d, "metadata.json")))
        pq = os.path.join(d, "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(d, "actions_raw.parquet")
        a = load_chunk_actions(pq)
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 303, 18, 2)
        if rc:
            print(f"[{m.get('game','?')[:18]:18s}] {summarize_chunk(rc)}")
