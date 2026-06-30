"""Browser state-authoring server for in-process emulator RL envs.

Start one RetroRLEnv, play it from a normal browser over an SSH/VS Code forwarded
port, and save frame-exact libretro states with human labels for later RL/eval
seeding.

Example:
  env -u VIRTUAL_ENV -u PYTHONPATH \
    PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc \
    CUDA_VISIBLE_DEVICES="" .venv/bin/python planner_poc/emulator_state_authoring_server.py \
      --rom "Game data/Sonic The Hedgehog 2.md" \
      --game SonicTheHedgehog2-Genesis-v0 --system Genesis --reward-var screen_x --port 8124

Forward the port (VS Code Ports tab, or ssh -L 9124:localhost:8124 host) and open
http://localhost:9124 in a local browser.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

REPO = Path(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "planner_poc"))

from nitrogen.eval.core import JLX, JLY  # noqa: E402
from nitrogen.eval.envs.retro_rl_env import RetroRLEnv  # noqa: E402
from nitrogen.shared import BUTTON_ACTION_TOKENS  # noqa: E402

ADIM = 25
NEUTRAL = 0.5
B = {n: i for i, n in enumerate(BUTTON_ACTION_TOKENS)}

# Browser KeyboardEvent.key lowercased -> NitroGen action effects.
KEYMAP = {
    "arrowleft": [("stick", JLX, 0.0), ("btn", B["DPAD_LEFT"])],
    "a": [("stick", JLX, 0.0), ("btn", B["DPAD_LEFT"])],
    "arrowright": [("stick", JLX, 1.0), ("btn", B["DPAD_RIGHT"])],
    "d": [("stick", JLX, 1.0), ("btn", B["DPAD_RIGHT"])],
    "arrowup": [("stick", JLY, 0.0), ("btn", B["DPAD_UP"])],
    "w": [("stick", JLY, 0.0), ("btn", B["DPAD_UP"])],
    "arrowdown": [("stick", JLY, 1.0), ("btn", B["DPAD_DOWN"])],
    "s": [("stick", JLY, 1.0), ("btn", B["DPAD_DOWN"])],
    "z": [("btn", B["SOUTH"])],
    " ": [("btn", B["SOUTH"])],
    "x": [("btn", B["WEST"])],
    "c": [("btn", B["EAST"])],
    "v": [("btn", B["NORTH"])],
    "q": [("btn", B["LEFT_SHOULDER"])],
    "e": [("btn", B["RIGHT_SHOULDER"])],
    "1": [("btn", B["LEFT_TRIGGER"])],
    "3": [("btn", B["RIGHT_TRIGGER"])],
    "enter": [("btn", B["START"])],
    "shift": [("btn", B["BACK"])],
}
KEYHELP = (
    "Arrows/WASD = move · Z/Space = A(jump) · X = B/attack · C = EAST · V = NORTH · "
    "Q/E = shoulders · 1/3 = triggers · Enter = START · Tab = one step"
)


def build_action(keys: set[str]) -> np.ndarray:
    a = np.zeros(ADIM, dtype=np.float32)
    a[JLX] = NEUTRAL
    a[JLY] = NEUTRAL
    for k in keys:
        for eff in KEYMAP.get(k, []):
            if eff[0] == "stick":
                a[eff[1]] = eff[2]
            else:
                a[eff[1]] = 1.0
    return a


def sanitize_label(label: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", (label or "").strip()).strip("._-")
    return label[:80] or datetime.now().strftime("snapshot_%Y%m%d_%H%M%S")


def json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


class EmulatorSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.env = RetroRLEnv(
            rom_path=args.rom,
            game=args.game,
            system=args.system,
            reward_var=args.reward_var,
            frames_per_row=args.frames_per_row,
            boot_start_frames=args.boot_start_frames,
            max_rows=10**9,
            max_stuck=10**9,
            done_on_life_zero=False,
        )
        self.game = args.game
        self.system = args.system
        self.reward_var = args.reward_var
        self.tick = float(args.tick)
        self.keys: set[str] = set()
        self.lock = threading.Lock()
        self.env_lock = threading.Lock()
        self.latest_jpeg = b""
        self.latest_info: dict = {}
        self.frame_id = 0
        self.frame_count = 0
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.fps = 0.0
        self.status = "booting"
        self.running = True
        self.mode = "step"
        self.autostep = True
        self.speed = 1.0
        self._step_req = 0
        self._last_frame_wall = time.time()
        self.state_dir = REPO / "tmp" / "states" / self.game
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def _reward_value_unlocked(self) -> float:
        try:
            return float(self.env._var(self.reward_var))
        except Exception:
            return 0.0

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        im = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        w, h = im.size
        maxw = 768
        if w > maxw:
            im = im.resize((maxw, max(1, round(h * maxw / w))), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=78)
        return buf.getvalue()

    def _publish_frame(self, frame: np.ndarray, info: dict | None = None) -> None:
        jpg = self._encode_frame(frame)
        rv = float((info or {}).get(self.reward_var, self._reward_value_unlocked()))
        with self.lock:
            self.latest_jpeg = jpg
            self.latest_info = dict(info or {})
            self.latest_info[self.reward_var] = rv
            self.latest_info.update(
                frame_count=self.frame_count,
                cumulative_reward=round(self.cumulative_reward, 6),
                last_reward=round(self.last_reward, 6),
                done=self.done,
            )
            self.frame_id += 1
            self._last_frame_wall = time.time()

    def _sync_reward_baseline_unlocked(self) -> None:
        rv = self._reward_value_unlocked()
        self.env._last_reward_val = rv
        if getattr(self.env, "score_var", None):
            self.env._last_score = self.env._var(self.env.score_var)
        self.env._stuck_count = 0

    def reset(self):
        self.status = "resetting"
        with self.lock:
            self.keys = set()
        with self.env_lock:
            frame = self.env.reset()
            self.frame_count = 0
            self.cumulative_reward = 0.0
            self.last_reward = 0.0
            self.done = False
            self._sync_reward_baseline_unlocked()
            info = {self.reward_var: self._reward_value_unlocked(), "reset": True}
            self._publish_frame(frame, info)
        self.status = "ready"

    def request_step(self, n: int = 1):
        self._step_req += max(1, int(n))

    def set_mode(self, mode: str):
        if mode in ("step", "realtime"):
            self.mode = mode

    def set_speed(self, speed: str | float):
        try:
            self.speed = max(0.05, min(4.0, float(speed)))
        except Exception:
            pass

    def set_autostep(self, on: bool):
        self.autostep = bool(on)

    def _advance_once(self, action: np.ndarray) -> None:
        frame, reward, done, info = self.env.step(action)
        self.frame_count += 1
        self.last_reward = float(reward)
        self.cumulative_reward += float(reward)
        self.done = bool(done)
        self._publish_frame(frame, info)

    def loop(self):
        while self.running:
            t0 = time.time()
            try:
                with self.lock:
                    keys = set(self.keys)
                should_step = self.mode == "realtime" or self._step_req > 0 or (self.mode == "step" and self.autostep and bool(keys))
                if should_step:
                    if self.mode == "step" and self._step_req > 0:
                        self._step_req -= 1
                    action = build_action(keys)
                    with self.env_lock:
                        self._advance_once(action)
                elif time.time() - self._last_frame_wall > 0.5:
                    with self.env_lock:
                        self._publish_frame(self.env.frame(), {self.reward_var: self._reward_value_unlocked()})
                dt = time.time() - t0
                if self.mode == "realtime":
                    game_dt = max(1.0 / max(float(getattr(self.env, "fps", 60.0)), 1.0), self.env.frames_per_row / max(float(getattr(self.env, "fps", 60.0)), 1.0))
                    target = game_dt / self.speed
                else:
                    target = self.tick
                if dt < target:
                    time.sleep(target - dt)
                period = time.time() - t0
                self.fps = round(1.0 / period, 1) if period > 0 else 0.0
            except Exception as e:
                self.status = f"error: {str(e)[:160]}"
                time.sleep(0.2)

    def list_snapshots(self):
        out = []
        for p in sorted(self.state_dir.glob("*.state")):
            meta = p.with_suffix(".json")
            item = {"label": p.stem, "state": str(p.relative_to(REPO)), "bytes": p.stat().st_size}
            if meta.exists():
                try:
                    m = json.loads(meta.read_text())
                    item.update(timestamp=m.get("timestamp"), reward_value_at_snapshot=m.get("reward_value_at_snapshot"), frame_count=m.get("frame_count"))
                except Exception:
                    pass
            out.append(item)
        return out

    def snapshot(self, label: str):
        safe = sanitize_label(label)
        state_path = self.state_dir / f"{safe}.state"
        meta_path = self.state_dir / f"{safe}.json"
        with self.env_lock:
            state = self.env.save_state()
            reward_value = self._reward_value_unlocked()
            frame = self.frame_count
        state_path.write_bytes(state)
        meta = {
            "game": self.game,
            "system": self.system,
            "rom_path": self.args.rom,
            "reward_var": self.reward_var,
            "reward_value_at_snapshot": reward_value,
            "frame_count": frame,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_bytes(json_bytes(meta))
        return {
            "ok": True,
            "label": safe,
            "state_path": str(state_path.relative_to(REPO)),
            "json_path": str(meta_path.relative_to(REPO)),
            "bytes": len(state),
            "snapshots": self.list_snapshots(),
        }

    def load(self, label: str):
        safe = sanitize_label(label)
        path = self.state_dir / f"{safe}.state"
        if not path.exists():
            raise FileNotFoundError(f"snapshot not found: {safe}")
        with self.lock:
            self.keys = set()
        with self.env_lock:
            self.env.load_state(path.read_bytes())
            self._sync_reward_baseline_unlocked()
            self.frame_count = 0
            self.cumulative_reward = 0.0
            self.last_reward = 0.0
            self.done = False
            frame = self.env.frame()
            self._publish_frame(frame, {self.reward_var: self._reward_value_unlocked(), "loaded": safe})
        return {"ok": True, "label": safe, "path": str(path.relative_to(REPO)), "snapshots": self.list_snapshots()}

    def status_dict(self):
        with self.lock:
            info = dict(self.latest_info)
            keys = sorted(self.keys)
        return {
            "ok": True,
            "game": self.game,
            "system": self.system,
            "reward_var": self.reward_var,
            "reward_value": info.get(self.reward_var, 0.0),
            "frame_count": self.frame_count,
            "cumulative_reward": round(self.cumulative_reward, 6),
            "last_reward": round(self.last_reward, 6),
            "done": self.done,
            "fps": self.fps,
            "mode": self.mode,
            "speed": self.speed,
            "autostep": self.autostep,
            "status": self.status,
            "keys": keys,
            "info": info,
            "snapshots": self.list_snapshots(),
        }

    def close(self):
        self.running = False
        time.sleep(0.2)
        with self.env_lock:
            try:
                self.env.close()
            except Exception:
                pass


SESSION: EmulatorSession | None = None

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>NitroGen emulator state authoring</title>
<style>body{background:#111;color:#ddd;font-family:monospace;margin:0;padding:12px}
img{image-rendering:pixelated;border:1px solid #333;max-width:96vw}button{font-family:monospace;font-size:14px;padding:6px 12px;margin:2px;background:#222;color:#ddd;border:1px solid #444;cursor:pointer}
button.on{background:#7a1d1d;border-color:#c33}select,input{font-family:monospace;font-size:14px;padding:5px;background:#222;color:#ddd;border:1px solid #444}#st{color:#9cf;white-space:pre-wrap}#hud{color:#888;font-size:12px}.blue{background:#1d4d7a;border-color:#39c}</style></head><body>
<h3 style="margin:0 0 8px 0">NitroGen emulator state authoring</h3>
<div><img id=v src="/stream"></div>
<div style=margin-top:8px><b>mode:</b>
<button id=m_step class=on onclick="setMode('step')">⏯ Step</button><button id=m_real onclick="setMode('realtime')">▶ Realtime</button>
<button id=m_auto class=on onclick="toggleAuto()" title="Step mode: hold a key to walk; turn off for Tab-only nudges">↻ auto-step</button>
<button class=blue onclick="doStep(1)">▶ Step (Tab)</button><button onclick="doStep(5)">+5</button><button onclick="doStep(30)">+30</button>
&nbsp;<b>speed:</b><button onclick="setSpeed(0.25)">0.25x</button><button onclick="setSpeed(0.5)">0.5x</button><button onclick="setSpeed(1)">1x</button><button onclick="setSpeed(2)">2x</button></div>
<div style=margin-top:8px><button onclick=doSnapshot() class=blue>📸 Snapshot State</button><select id=snaps></select><button onclick=doLoad()>Load</button><button onclick=doReset()>↺ Reset</button><span id=msg></span></div>
<div id=st></div><div id=hud>__KEYHELP__<br>Snapshots are written under <b>tmp/states/&lt;game&gt;/</b>. Use normal local browser focus; blur clears held keys.</div>
<script>
let held=new Set(), mode='step';
function send(){fetch('/keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys:[...held]})});}
function setMode(m){mode=m;fetch('/mode',{method:'POST',body:m}).then(()=>{document.getElementById('m_step').className=(m=='step')?'on':'';document.getElementById('m_real').className=(m=='realtime')?'on':'';});}
function setSpeed(s){fetch('/speed',{method:'POST',body:''+s});}
function toggleAuto(){let b=document.getElementById('m_auto');let on=b.className!='on';b.className=on?'on':'';fetch('/autostep',{method:'POST',body:on?'1':'0'});}
function doStep(n){fetch('/step',{method:'POST',body:''+(n||1)});}
function doReset(){held.clear();send();fetch('/reset',{method:'POST'});}
function refreshSnaps(snaps){let s=document.getElementById('snaps');let old=s.value;s.innerHTML='';(snaps||[]).forEach(x=>{let o=document.createElement('option');o.value=x.label;o.textContent=x.label+' ('+x.bytes+' bytes)';if(x.label==old)o.selected=true;s.appendChild(o);});}
function doSnapshot(){let label=prompt('Snapshot label (e.g. emerald_hill_1_start)','');if(!label)return;fetch('/snapshot',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label})}).then(r=>r.json()).then(j=>{document.getElementById('msg').textContent=j.ok?' saved '+j.state_path:(' error '+j.error);refreshSnaps(j.snapshots);});}
function doLoad(){let label=document.getElementById('snaps').value;if(!label)return;held.clear();send();fetch('/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label})}).then(r=>r.json()).then(j=>{document.getElementById('msg').textContent=j.ok?' loaded '+j.path:(' error '+j.error);refreshSnaps(j.snapshots);});}
addEventListener('keydown',e=>{if(e.target.tagName=='INPUT'||e.target.tagName=='SELECT')return;let k=e.key.toLowerCase();if(k=='tab'){e.preventDefault();doStep(1);return;}if(k.startsWith('arrow')||['z','x','c','v','q','e','w','a','s','d','1','3','enter',' ','shift'].includes(k)){e.preventDefault();if(document.activeElement&&document.activeElement.blur)document.activeElement.blur();if(!held.has(k)){held.add(k);send();}}});
addEventListener('keyup',e=>{let k=e.key.toLowerCase();if(held.delete(k))send();});addEventListener('blur',()=>{held.clear();send();});
setInterval(()=>fetch('/status').then(r=>r.json()).then(j=>{document.getElementById('st').textContent='game: '+j.game+'  status:'+j.status+'  mode:'+j.mode+(j.mode=='step'?(' auto:'+(j.autostep?'on':'off')):(' speed:'+j.speed+'x'))+'\n'+j.reward_var+': '+j.reward_value+'  cum_reward:'+j.cumulative_reward+'  frame_count:'+j.frame_count+'  fps:'+j.fps+'  done:'+j.done+'\ninfo: '+JSON.stringify(j.info);refreshSnaps(j.snapshots);document.getElementById('m_auto').className=j.autostep?'on':'';}),500);
</script></body></html>""".replace("__KEYHELP__", KEYHELP)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code=200, body=b"", ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _body_text(self) -> str:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n).decode("utf-8", "replace") if n else ""

    def _body_json(self):
        body = self._body_text()
        if not body:
            return {}
        try:
            return json.loads(body)
        except Exception:
            return body

    def do_GET(self):
        global SESSION
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode(), "text/html")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            last_id = -1
            try:
                while SESSION and SESSION.running:
                    with SESSION.lock:
                        jpg = SESSION.latest_jpeg
                        fid = SESSION.frame_id
                    if jpg and fid != last_id:
                        last_id = fid
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    else:
                        time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif path == "/status":
            self._send(200, json.dumps(SESSION.status_dict()).encode(), "application/json")
        elif path == "/snapshots":
            self._send(200, json.dumps({"snapshots": SESSION.list_snapshots()}).encode(), "application/json")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        global SESSION
        path = urlparse(self.path).path
        try:
            if path == "/keys":
                data = self._body_json()
                if isinstance(data, dict):
                    keys = data.get("keys", [])
                elif isinstance(data, list):
                    keys = data
                else:
                    keys = []
                with SESSION.lock:
                    SESSION.keys = {str(k).lower() for k in keys}
                self._send(200, b"ok")
            elif path == "/step":
                try:
                    n = int(str(self._body_text()).strip() or "1")
                except Exception:
                    n = 1
                SESSION.request_step(n)
                self._send(200, b"ok")
            elif path == "/mode":
                SESSION.set_mode(self._body_text().strip())
                self._send(200, json.dumps({"mode": SESSION.mode}).encode(), "application/json")
            elif path == "/speed":
                SESSION.set_speed(self._body_text().strip())
                self._send(200, json.dumps({"speed": SESSION.speed}).encode(), "application/json")
            elif path == "/autostep":
                SESSION.set_autostep(self._body_text().strip() not in ("0", "false", ""))
                self._send(200, json.dumps({"autostep": SESSION.autostep}).encode(), "application/json")
            elif path == "/reset":
                threading.Thread(target=SESSION.reset, daemon=True).start()
                self._send(200, b"ok")
            elif path == "/snapshot":
                data = self._body_json()
                label = data.get("label", "") if isinstance(data, dict) else str(data)
                self._send(200, json.dumps(SESSION.snapshot(label)).encode(), "application/json")
            elif path == "/load":
                data = self._body_json()
                label = data.get("label", "") if isinstance(data, dict) else str(data)
                self._send(200, json.dumps(SESSION.load(label)).encode(), "application/json")
            else:
                self._send(404, b"not found")
        except Exception as e:
            self._send(500, json.dumps({"ok": False, "error": str(e)}).encode(), "application/json")


def main():
    global SESSION
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="Game data/Sonic The Hedgehog 2.md")
    ap.add_argument("--game", default="SonicTheHedgehog2-Genesis-v0")
    ap.add_argument("--system", default="Genesis")
    ap.add_argument("--reward-var", default="screen_x")
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--tick", type=float, default=0.066, help="step-mode polling period in seconds")
    ap.add_argument("--frames-per-row", type=int, default=4)
    ap.add_argument("--boot-start-frames", type=int, default=600)
    args = ap.parse_args()

    print(f"[emu-state] booting {args.game} rom={args.rom} reward={args.reward_var}", flush=True)
    SESSION = EmulatorSession(args)
    threading.Thread(target=SESSION.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)

    def _shutdown(*_):
        try:
            if SESSION:
                SESSION.close()
        finally:
            try:
                srv.server_close()
            finally:
                os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    print(f"[emu-state] serving on http://localhost:{args.port} (forward this port; open in your browser)", flush=True)
    print(f"[emu-state] keys: {KEYHELP}", flush=True)
    try:
        srv.serve_forever()
    finally:
        if SESSION:
            SESSION.close()
        srv.server_close()


if __name__ == "__main__":
    main()
