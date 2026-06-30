"""Boot-test every recordable env and report what actually works RIGHT NOW.

For each env name from run_poc.list_envs(): build + boot it (freeze=True, like the record server),
grab one frame, and classify:
  OK        - booted and produced a non-black frame
  BLANK     - booted but the frame is black/near-black (render/focus problem)
  FAIL:<e>  - construction/boot raised (missing artifact, GL error, ...)
  TIMEOUT   - exceeded the per-env wall-clock budget

Each env runs in a SUBPROCESS so a hang/segfault can't take down the harness; we hard-kill on timeout.
Usage: env_healthcheck.py [--only a,b,c] [--timeout 45]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")


def _probe_one(name: str) -> dict:
    """Runs in a child process: build the env, grab a frame, print a JSON verdict."""
    import numpy as np
    sys.path[:0] = [REPO, os.path.join(REPO, "planner_poc")]
    from run_poc import make_env_factory
    out = {"env": name}
    env = None
    try:
        env = make_env_factory(name, freeze=True)()
        # grab a frame via whatever the env supports
        frame = None
        if hasattr(env, "_grab"):
            frame = env._grab()
        elif hasattr(env, "apply_chunk_capture"):
            import numpy as _np
            a = _np.zeros((25,), _np.float32); a[21] = a[22] = 0.5
            try:
                rows = env.apply_chunk_capture(a[None], per_row=0.05)
            except TypeError:
                rows = env.apply_chunk_capture(a[None])
            frame = rows[-1][1]
        if frame is None:
            out["status"] = "FAIL:no-grab"
        else:
            arr = np.asarray(frame, dtype=np.float32)
            out["std"] = round(float(arr.std()), 1)
            out["status"] = "OK" if arr.std() > 3.0 else "BLANK"
        out["control"] = getattr(env, "control", "?")
    except Exception as e:
        out["status"] = "FAIL:" + type(e).__name__ + ":" + str(e).split("\n")[0][:80]
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=float, default=50.0)
    ap.add_argument("--probe", default="")   # internal: run a single env probe and print JSON
    args = ap.parse_args()

    if args.probe:
        print("RESULT " + json.dumps(_probe_one(args.probe)), flush=True)
        return

    sys.path[:0] = [REPO, os.path.join(REPO, "planner_poc")]
    from run_poc import list_envs
    names = [s.strip() for s in args.only.split(",") if s.strip()] or list_envs()

    base_env = dict(os.environ)
    base_env.pop("VIRTUAL_ENV", None); base_env.pop("PYTHONPATH", None)
    base_env["PYTHONPATH"] = f"{REPO}:{REPO}/planner_poc"
    base_env.setdefault("QWEN", "Qwen/Qwen3.5-2B")

    results = []
    for nm in names:
        cmd = [sys.executable, __file__, "--probe", nm]
        try:
            p = subprocess.run(cmd, env=base_env, capture_output=True, text=True,
                               timeout=args.timeout)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line:
                res = json.loads(line[len("RESULT "):])
            else:
                err = (p.stderr.strip().splitlines() or ["?"])[-1][:90]
                res = {"env": nm, "status": "FAIL:nooutput:" + err}
        except subprocess.TimeoutExpired:
            res = {"env": nm, "status": "TIMEOUT"}
        results.append(res)
        print(f"{res['status']:<34} {nm:<24} {('std=' + str(res.get('std'))) if 'std' in res else ''}",
              flush=True)

    print("\n==== SUMMARY ====")
    for tag in ("OK", "BLANK", "TIMEOUT"):
        hits = [r["env"] for r in results if r["status"].startswith(tag)]
        print(f"{tag} ({len(hits)}): {', '.join(hits)}")
    fails = [r for r in results if r["status"].startswith("FAIL")]
    print(f"FAIL ({len(fails)}):")
    for r in fails:
        print(f"   {r['env']:<24} {r['status']}")
    with open("/tmp/env_healthcheck.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nfull results -> /tmp/env_healthcheck.json")


if __name__ == "__main__":
    main()
