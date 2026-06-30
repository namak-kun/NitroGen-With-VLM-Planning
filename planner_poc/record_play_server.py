"""Browser play-and-record server: play any NitroGen GameEnv LIVE in your laptop browser (over an SSH /
VS Code forwarded port) and record aligned GOLD (frame, action, state) trajectories by playing.

WHY: the machine is headless (Xvfb, no local display). This serves the game's frames to your browser as
an MJPEG stream and sends your keypresses back, so you play with your real keyboard and every step is
logged as a NitroGen 25-dim action paired with the frame the policy would see -- clean gold data.

HOW TO CONNECT:
  1. Start it on the remote box:  (example: SuperTuxKart)
       env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. .venv/bin/python planner_poc/record_play_server.py \
           --env stk --port 8080
  2. Forward the port to your laptop:
       - VS Code Remote-SSH: a "port 8080 available" popup appears -> Open in Browser (or the Ports tab).
       - plain SSH:  ssh -L 8080:localhost:8080 user@host
  3. Open http://localhost:8080 in your NORMAL local browser (not the VS Code webview). Play.

CONTROLS (default keymap, configurable): Arrow keys / WASD = move (left stick), Z = SOUTH (A/jump),
  X = WEST (B/attack), C = EAST, V = NORTH, A/S held = shoulders, Enter = START, Space = SOUTH too.
UI buttons: Record (toggle logging), Reset (re-run the env reset macro), Mark (insert an event marker),
  and a Task field (annotate the current recording with a short task like "get the flower, go right").

OUTPUT: recordings go to docs/recordings/<env>_<timestamp>/  ->  frames/NNNNNN.png + actions.jsonl
  (one JSON line per recorded step: idx, t, action [25 floats], buttons, stick, keys, state, task, mark).
  Convert to the training action layout later; this keeps the raw aligned (frame, action, state).
"""
import argparse
import io
import json
import os
import sys
import signal
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.eval import Scenario
from nitrogen.eval.core import JLX, JLY
from nitrogen.shared import BUTTON_ACTION_TOKENS

# NitroGen 25-dim action: 0..20 buttons (BUTTON_ACTION_TOKENS), 21/22 left stick, 23/24 right stick.
ADIM = 25
B = {n: i for i, n in enumerate(BUTTON_ACTION_TOKENS)}
NEUTRAL = 0.5

# browser key (KeyboardEvent.key, lowercased) -> effect. ("stick", axis, value) or ("btn", idx).
# sticks are [0,1], 0.5 neutral; we also set the matching DPAD bit so dpad-reading envs respond.
KEYMAP = {
    "arrowleft": [("stick", JLX, 0.0), ("btn", B["DPAD_LEFT"])],
    "a":         [("stick", JLX, 0.0), ("btn", B["DPAD_LEFT"])],
    "arrowright": [("stick", JLX, 1.0), ("btn", B["DPAD_RIGHT"])],
    "d":         [("stick", JLX, 1.0), ("btn", B["DPAD_RIGHT"])],
    "arrowup":   [("stick", JLY, 0.0), ("btn", B["DPAD_UP"])],
    "w":         [("stick", JLY, 0.0), ("btn", B["DPAD_UP"])],
    "arrowdown": [("stick", JLY, 1.0), ("btn", B["DPAD_DOWN"])],
    "s":         [("stick", JLY, 1.0), ("btn", B["DPAD_DOWN"])],
    "z":         [("btn", B["SOUTH"])],     # A / jump / confirm
    " ":         [("btn", B["SOUTH"])],     # space = SOUTH too
    "x":         [("btn", B["WEST"])],      # B / attack / shoot
    "c":         [("btn", B["EAST"])],
    "v":         [("btn", B["NORTH"])],
    "q":         [("btn", B["LEFT_SHOULDER"])],
    "e":         [("btn", B["RIGHT_SHOULDER"])],
    "1":         [("btn", B["LEFT_TRIGGER"])],
    "3":         [("btn", B["RIGHT_TRIGGER"])],
    "enter":     [("btn", B["START"])],
    "shift":     [("btn", B["BACK"])],
}
KEYHELP = ("Arrows/WASD = move · Z/Space = A(jump) · X = B(attack) · C = EAST · V = NORTH · "
           "Q/E = shoulders · 1/3 = triggers · Enter = START")


def build_action(keys: set) -> np.ndarray:
    a = np.full(ADIM, 0.0, dtype=np.float32)
    a[JLX] = NEUTRAL; a[JLY] = NEUTRAL
    for k in keys:
        for eff in KEYMAP.get(k, []):
            if eff[0] == "stick":
                a[eff[1]] = eff[2]
            else:
                a[eff[1]] = 1.0
    return a


def make_env(name, boot_wait=None, freeze=True):
    # freeze=True: boot WITH the SpeedHack time-shim so the server can control game speed live --
    # run real-time at any scale (speed slider) OR hold the world frozen and advance N frames per
    # keypress (step mode). Both kill the tunnel input-lag that makes continuous real-time play twitchy.
    from run_poc import make_env_factory
    kw = {"freeze": freeze}
    if boot_wait is not None:
        kw["boot_wait"] = boot_wait
    return make_env_factory(name, **kw)()


class GameSession:
    """Runs the env in a background thread: each tick applies the held keys for one short slice, grabs a
    frame + state, and (if recording) logs the aligned (frame, action, state)."""

    def __init__(self, builder, env_name, tick=0.12):
        self.builder = builder           # builder(name) -> a booted GameEnv
        self.env = builder(env_name)
        self.env_name = env_name
        self.tick = tick
        self.keys = set()
        self.lock = threading.Lock()
        self.env_lock = threading.Lock()  # guards env swap vs the game loop
        self.latest_jpeg = b""
        self.latest_state = {}
        self.frame_id = 0
        self.running = True
        self.recording = False
        self.rec_dir = None
        self.rec_idx = 0
        self.task = ""
        self._mark = None
        self.fps = 0.0
        self.status = "ready"            # ready | booting <env> | error: ...
        self.H = getattr(getattr(self.env, "config", None), "action_horizon", 18) or 18
        self._mss = None                 # lazily-created fast-grab handle (per loop thread)
        self._mss_disp = None            # display the current _mss is bound to
        # --- play modes (both kill tunnel input-lag) -------------------------------------
        # "step"     : world FROZEN; you set up held keys, then tap Tab/▶ to advance step_seconds of
        #              game time with those keys applied -> zero lag, exact (frame,action) alignment.
        # "realtime" : world runs continuously at `speed`x (0.25/0.5/1x/2x). Slower = far less twitchy
        #              under lag. Held keys persist (xdotool).
        self.mode = "step"               # default to the lag-free mode for clean gold recording
        self.speed = 0.5                 # realtime speed scale (slider)
        self.step_seconds = 0.12         # game-time advanced per committed step
        self._step_req = 0               # pending step commits (set by /step)
        self.autostep = False            # step mode: OFF = deterministic (1 Tab = 1 move, over-input-proof);
                                         # ON = hold a key to walk (smoother for platformers)
        self._switch_gen = 0             # monotonically increments per switch request; only the latest wins
        self._applied_scale = None       # last scale pushed to the SpeedHack (avoid redundant writes)

    def _set_scale(self, env, scale):
        sh = getattr(env, "_sh", None)
        if sh is not None and scale != self._applied_scale:
            try:
                sh.set_speed(scale); self._applied_scale = scale
            except Exception:
                pass

    def _fast_grab(self, env):
        """Capture a frame via mss (in-process X11, ~0.6ms) instead of spawning ffmpeg (~170ms).
        Falls back to env._grab() on any failure. Must be called only from the loop thread (mss holds
        a per-thread X connection)."""
        disp = getattr(env, "display", None)
        if disp is None:
            return env._grab()
        try:
            import mss
            if self._mss is None or self._mss_disp != disp:
                if self._mss is not None:
                    try: self._mss.close()
                    except Exception: pass
                self._mss = mss.MSS(display=f":{disp}")
                self._mss_disp = disp
            mon = {"left": 0, "top": 0, "width": env.width, "height": env.height}
            raw = self._mss.grab(mon)
            arr = np.asarray(raw)            # BGRA, (H,W,4)
            return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])  # -> RGB
        except Exception:
            self._mss = None; self._mss_disp = None
            return env._grab()

    def switch_env(self, name):
        """Tear down the current env and boot a NEW one in place -- the browser/port stay connected,
        only the game changes. SERIALIZED: each call bumps a generation counter; if newer switch
        requests arrive while this one is still booting (e.g. you accidentally scrolled the dropdown
        through several envs), the stale ones bail out instead of booting many games at once."""
        self._switch_gen += 1
        my_gen = self._switch_gen
        def _do():
            if self.recording:
                self.stop_recording()
            if my_gen != self._switch_gen:
                return                                  # superseded before we even started
            self.status = f"booting {name} ..."
            with self.lock:
                self.keys = set()
            with self.env_lock:
                if my_gen != self._switch_gen:          # superseded while waiting for the lock
                    return
                old = self.env
                self.env = None
                try:
                    old.close()
                except Exception:
                    pass
                try:
                    self.env = self.builder(name)
                    self.env_name = name
                    self.H = getattr(getattr(self.env, "config", None), "action_horizon", 18) or 18
                    self.status = "ready"
                except Exception as e:
                    self.status = f"error: {str(e)[:120]}"
                    return
            if my_gen == self._switch_gen:
                self.reset()
        threading.Thread(target=_do, daemon=True).start()

    def reset(self):
        self.status = "resetting ..."     # give the UI immediate feedback (macro replay can be slow)
        with self.lock:
            self.keys = set()
        try:
            with self.env_lock:
                if self.env is None:
                    return
                sc = Scenario(self.env_name, plan="", objective="play", max_steps=10 ** 9)
                obs = self.env.reset(sc)
                self._disable_autorepeat(self.env)
            self._applied_scale = None     # env.reset left the SpeedHack paused; let the loop re-apply
            self._store_frame(obs.frame, obs.state if isinstance(obs.state, dict) else {})
        finally:
            self.status = "ready"

    @staticmethod
    def _disable_autorepeat(env):
        """Turn off X key autorepeat on the env's display so a held key can't inject phantom repeat
        keydown events that pile up in the game's input queue (a source of over-input under lag)."""
        disp = getattr(env, "display", None)
        if disp is None:
            return
        try:
            subprocess.run(["xset", "-display", f":{disp}", "r", "off"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        except Exception:
            pass

    def request_step(self, n=1):
        self._step_req += max(1, int(n))

    def set_mode(self, mode):
        if mode in ("step", "realtime"):
            self.mode = mode
            self._applied_scale = None     # force the loop to re-apply the right scale for this mode

    def set_speed(self, speed):
        try:
            self.speed = max(0.03, min(4.0, float(speed)))
            self._applied_scale = None
        except Exception:
            pass

    def set_autostep(self, on):
        self.autostep = bool(on)

    def _store_frame(self, frame, state):
        # Downscale ONLY the browser-stream JPEG (saves ~4x bandwidth over a forwarded/tunneled port so
        # play stays smooth). Recorded gold frames (_log_step) keep FULL resolution.
        im = Image.fromarray(np.asarray(frame)).convert("RGB")
        w, h = im.size
        maxw = 512
        if w > maxw:
            im = im.resize((maxw, max(1, round(h * maxw / w))), Image.BILINEAR)
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=72)
        with self.lock:
            self.latest_jpeg = buf.getvalue()
            self.latest_state = state or {}
            self.frame_id += 1

    def start_recording(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rec_dir = os.path.join(REPO, "docs", "recordings", f"{self.env_name}_{ts}")
        os.makedirs(os.path.join(self.rec_dir, "frames"), exist_ok=True)
        self.rec_idx = 0
        self.recording = True
        with open(os.path.join(self.rec_dir, "meta.json"), "w") as f:
            json.dump({"env": self.env_name, "started": ts, "task": self.task,
                       "action_layout": "0..20 buttons (BUTTON_ACTION_TOKENS), 21/22 left stick [0,1] "
                       "0.5=neutral, 23/24 right stick", "keymap_help": KEYHELP}, f, indent=2)
        return self.rec_dir

    def stop_recording(self):
        self.recording = False
        return self.rec_dir, self.rec_idx

    def _log_step(self, action, frame, state):
        idx = self.rec_idx
        Image.fromarray(np.asarray(frame)).convert("RGB").save(
            os.path.join(self.rec_dir, "frames", f"{idx:06d}.png"))
        btns = [BUTTON_ACTION_TOKENS[i] for i in range(21) if action[i] > 0.5]
        rec = {"idx": idx, "t": round(time.time(), 3), "action": [round(float(x), 3) for x in action],
               "buttons": btns, "stick": [round(float(action[JLX]), 3), round(float(action[JLY]), 3)],
               "keys": sorted(self.keys), "state": {k: v for k, v in (state or {}).items()},
               "task": self.task}
        if self._mark:
            rec["mark"] = self._mark; self._mark = None
        with open(os.path.join(self.rec_dir, "actions.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.rec_idx += 1

    def loop(self):
        """Human-play loop. Game time is advanced SYNCHRONOUSLY in fixed slices locked to the frames we
        deliver -- the game is frozen except during a controlled unfreeze, so network/tunnel lag becomes
        pure latency (you see frames late) and can NEVER make the game run free and burst-catch-up (the
        "actions collected then executed" feel). Two modes:

        REALTIME: every tick, hold the current keys, unfreeze for (tick * speed) seconds of GAME time,
        refreeze, grab. speed<1 = lag-immune slow-mo. Recording logs every advanced frame.

        STEP: frozen until you advance. Advance happens on a Tab/▶ commit, OR (autostep, default ON)
        automatically whenever a key is held -- so holding Right just walks, one logged frame per slice.
        Idle (no keys, no commit) stays frozen. Zero input lag, exact (frame, action) alignment.

        Non-keyboard envs (gamepad / in-process emulators that step on a call) use apply_chunk_capture,
        called defensively since some envs' signatures don't accept per_row."""
        last_env_keys = None
        while self.running:
            t0 = time.time()
            if self.env is None or self.status != "ready":
                last_env_keys = None; self._applied_scale = None
                time.sleep(0.1); continue
            with self.lock:
                keys = set(self.keys)
            action = build_action(keys)
            log_now = False
            try:
                with self.env_lock:
                    if self.env is None:
                        time.sleep(0.05); continue
                    env = self.env
                    is_kbd = getattr(env, "control", None) == "keyboard" and hasattr(env, "_set_keys")
                    if is_kbd:
                        intended = env.action_to_keys(action[None])   # keys the held input WANTS
                        # decide how much GAME time to advance, AND which keys to actually forward to the
                        # game. In pure step mode while idle we forward NEUTRAL: your frantic direction
                        # taps (over-pressed because of lag) then can't queue up in the game's input
                        # buffer -- only the keys held AT a commit count, so #moves == #Tab presses.
                        if self.mode == "step":
                            if self._step_req > 0:                    # deliberate Tab/▶ nudge = ONE move
                                self._step_req -= 1
                                apply_keys = intended; game_dt = self.step_seconds
                            elif self.autostep and bool(intended):    # hold a key -> walk (lag-immune)
                                apply_keys = intended; game_dt = self.tick * self.speed
                            else:                                     # idle: frozen + NEUTRAL (no queueing)
                                apply_keys = set(); game_dt = 0.0
                        else:
                            apply_keys = intended; game_dt = self.tick * self.speed   # realtime slice
                        if apply_keys != last_env_keys:
                            env._set_keys(apply_keys); last_env_keys = apply_keys
                        if game_dt > 0:
                            self._set_scale(env, 1.0)                # run real-time for this slice
                            time.sleep(game_dt)
                            self._set_scale(env, 0.02)               # refreeze (lag can't advance it)
                            log_now = self.recording
                        else:
                            self._set_scale(env, 0.02)               # idle: keep frozen
                        frame = self._fast_grab(env)
                        try:
                            state = env.read_state()
                        except Exception:
                            state = {}
                    else:
                        try:
                            rows = env.apply_chunk_capture(action[None], per_row=self.tick)
                        except TypeError:
                            rows = env.apply_chunk_capture(action[None])   # envs w/o a per_row kwarg
                        _, frame, state = rows[-1]
                        log_now = self.recording
            except Exception as e:
                frame = np.zeros((240, 320, 3), np.uint8); state = {"error": str(e)[:80]}
            self._store_frame(frame, state)
            if self.recording and log_now:
                try:
                    self._log_step(action, frame, state)
                except Exception:
                    pass
            dt = time.time() - t0
            if dt < self.tick:                          # pace the stream (grab is now ~5ms, so we cap rate)
                time.sleep(self.tick - dt)
            period = time.time() - t0                    # true loop period incl. pacing sleep
            self.fps = round(1.0 / period, 1) if period > 0 else 0.0

    def close(self):
        self.running = False
        time.sleep(0.3)
        try:
            self.env.close()
        except Exception:
            pass


SESSION: GameSession = None
ENV_NAMES = []          # populated in main() from the factory; envs switchable from the browser

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>NitroGen record</title>
<style>body{background:#111;color:#ddd;font-family:monospace;margin:0;padding:12px}
img{image-rendering:pixelated;border:1px solid #333;max-width:96vw}
button{font-family:monospace;font-size:14px;padding:6px 12px;margin:2px;background:#222;color:#ddd;border:1px solid #444;cursor:pointer}
button.on{background:#7a1d1d;border-color:#c33}
select,#task{font-family:monospace;font-size:14px;padding:5px;background:#222;color:#ddd;border:1px solid #444}
#task{width:320px}#st{color:#9cf;white-space:pre}#hud{color:#888;font-size:12px}#env{color:#fc9}</style></head><body>
<div style=margin-bottom:6px>
game: <select id=envsel onchange=switchEnv()></select>
<span id=env></span></div>
<div><img id=v src="/stream"></div>
<div style=margin-top:8px>
<b>mode:</b>
<button id=m_step class=on onclick="setMode('step')">⏯ Step</button>
<button id=m_real onclick="setMode('realtime')">▶ Realtime</button>
<button id=m_auto onclick="toggleAuto()" title="Step mode: hold a key to walk (lag-immune, but less precise)">↻ auto-step</button>
&nbsp; <b>speed:</b>
<button class=spd onclick="setSpeed(0.0625)">0.06x</button>
<button class=spd onclick="setSpeed(0.125)">0.12x</button>
<button class=spd onclick="setSpeed(0.25)">0.25x</button>
<button class=spd onclick="setSpeed(0.5)">0.5x</button>
<button class=spd onclick="setSpeed(1)">1x</button>
<button class=spd onclick="setSpeed(2)">2x</button>
<button id=stepbtn onclick="doStep()" style="background:#1d4d7a;border-color:#39c">▶ Step (Tab)</button>
</div>
<div style=margin-top:8px>
<button id=rec onclick=toggleRec()>● Record</button>
<button onclick=doReset()>↺ Reset</button>
<button onclick=fetch('/mark',{method:'POST',body:prompt('marker label','event')||''})>⚑ Mark</button>
<input id=task placeholder="task label e.g. get the flower, go right" onchange=setTask()>
<span id=recinfo></span></div>
<div id=st></div><div id=hud>__KEYHELP__<br><b>Step</b> mode (lag-free): hold a key to walk (auto-step ON), or tap <b>Tab</b> to nudge one slice. <b>Realtime</b>: runs continuously at the chosen speed (game time is locked to frames, so lag never causes input bursts). Lower speed = easier under lag.</div>
<script>
let held=new Set(), rec=false, mode='step';
fetch('/envs').then(r=>r.json()).then(j=>{let s=document.getElementById('envsel');j.envs.forEach(e=>{let o=document.createElement('option');o.value=o.textContent=e;if(e==j.current)o.selected=true;s.appendChild(o);});});
function send(){fetch('/keys',{method:'POST',body:JSON.stringify([...held])});}
function doStep(){fetch('/step',{method:'POST',body:'1'});}
function doReset(){document.getElementById('env').textContent=' resetting...';fetch('/reset',{method:'POST'});}
function setMode(m){mode=m;fetch('/mode',{method:'POST',body:m}).then(()=>{document.getElementById('m_step').className=(m=='step')?'on':'';document.getElementById('m_real').className=(m=='realtime')?'on':'';document.getElementById('stepbtn').style.display=(m=='step')?'':'none';document.getElementById('m_auto').style.display=(m=='step')?'':'none';});}
function setSpeed(s){fetch('/speed',{method:'POST',body:''+s});}
function toggleAuto(){let b=document.getElementById('m_auto');let on=b.className!='on';b.className=on?'on':'';fetch('/autostep',{method:'POST',body:on?'1':'0'});}
addEventListener('keydown',e=>{if(e.target.tagName=='INPUT')return;let k=e.key.toLowerCase();if(k=='tab'){e.preventDefault();doStep();return;}if(k.startsWith('arrow')||['z','x','c','v','q','e','w','a','s','d','1','3','enter',' ','shift'].includes(k)){e.preventDefault();if(document.activeElement&&document.activeElement.blur)document.activeElement.blur();if(!held.has(k)){held.add(k);send();}}});
addEventListener('keyup',e=>{let k=e.key.toLowerCase();if(held.delete(k))send();});
addEventListener('blur',()=>{held.clear();send();});
function toggleRec(){fetch('/record',{method:'POST'}).then(r=>r.json()).then(j=>{rec=j.recording;let b=document.getElementById('rec');b.className=rec?'on':'';b.textContent=rec?'■ Stop':'● Record';document.getElementById('recinfo').textContent=rec?(' REC -> '+j.dir):(' saved '+j.n+' steps');});}
function setTask(){fetch('/task',{method:'POST',body:document.getElementById('task').value});}
function switchEnv(){held.clear();let s=document.getElementById('envsel');let e=s.value;s.blur();document.getElementById('env').textContent=' booting '+e+'...';fetch('/switch',{method:'POST',body:e});}
setInterval(()=>fetch('/status').then(r=>r.json()).then(j=>{document.getElementById('st').textContent='state: '+JSON.stringify(j.state)+'   fps:'+j.fps+'  mode:'+j.mode+(j.mode=='step'?(' auto:'+(j.autostep?'on':'off')):(' speed:'+j.speed+'x'))+(j.recording?('  REC '+j.n):'');document.getElementById('env').textContent=' ['+j.env+']'+(j.status!='ready'?('  '+j.status):'');let a=document.getElementById('m_auto');if(a)a.className=j.autostep?'on':'';}),500);
</script></body></html>""".replace("__KEYHELP__", KEYHELP)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code=200, body=b"", ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n).decode("utf-8", "replace") if n else ""

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            last_id = -1
            try:
                while SESSION and SESSION.running:
                    with SESSION.lock:
                        jpg = SESSION.latest_jpeg; fid = SESSION.frame_id
                    if jpg and fid != last_id:        # push ONLY fresh frames -> minimal display lag
                        last_id = fid
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg); self.wfile.write(b"\r\n")
                    else:
                        time.sleep(0.005)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/status":
            with SESSION.lock:
                st = dict(SESSION.latest_state)
            self._send(200, json.dumps({"state": st, "fps": SESSION.fps,
                       "recording": SESSION.recording, "n": SESSION.rec_idx,
                       "env": SESSION.env_name, "status": SESSION.status,
                       "mode": SESSION.mode, "speed": SESSION.speed,
                       "autostep": SESSION.autostep,
                       "step_seconds": SESSION.step_seconds}).encode(),
                       "application/json")
        elif self.path == "/envs":
            self._send(200, json.dumps({"envs": ENV_NAMES, "current": SESSION.env_name}).encode(),
                       "application/json")
        else:
            self._send(404)

    def do_POST(self):
        body = self._body()
        if self.path == "/keys":
            try:
                ks = set(json.loads(body))
            except Exception:
                ks = set()
            with SESSION.lock:
                SESSION.keys = ks
            self._send(200, b"ok")
        elif self.path == "/record":
            if SESSION.recording:
                d, n = SESSION.stop_recording()
                self._send(200, json.dumps({"recording": False, "n": n, "dir": d}).encode(),
                           "application/json")
            else:
                d = SESSION.start_recording()
                self._send(200, json.dumps({"recording": True, "n": 0, "dir": d}).encode(),
                           "application/json")
        elif self.path == "/reset":
            threading.Thread(target=SESSION.reset, daemon=True).start()
            self._send(200, b"ok")
        elif self.path == "/switch":
            name = body.strip()
            if name in ENV_NAMES:
                SESSION.switch_env(name)
                self._send(200, b"ok")
            else:
                self._send(400, b"unknown env")
        elif self.path == "/mode":
            SESSION.set_mode(body.strip())
            self._send(200, json.dumps({"mode": SESSION.mode}).encode(), "application/json")
        elif self.path == "/speed":
            SESSION.set_speed(body.strip())
            self._send(200, json.dumps({"speed": SESSION.speed}).encode(), "application/json")
        elif self.path == "/autostep":
            SESSION.set_autostep(body.strip() not in ("0", "false", ""))
            self._send(200, json.dumps({"autostep": SESSION.autostep}).encode(), "application/json")
        elif self.path == "/step":
            n = 1
            try: n = int(body.strip() or "1")
            except Exception: pass
            SESSION.request_step(n)
            self._send(200, b"ok")
        elif self.path == "/task":
            SESSION.task = body.strip()
            self._send(200, b"ok")
        elif self.path == "/mark":
            SESSION._mark = body.strip() or "mark"
            self._send(200, b"ok")
        else:
            self._send(404)


def main():
    global SESSION, ENV_NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="stk")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tick", type=float, default=0.033, help="target stream/control period in seconds (~30fps)")
    ap.add_argument("--boot-wait", type=float, default=None)
    args = ap.parse_args()

    from run_poc import list_envs
    ENV_NAMES = list_envs()
    builder = (lambda nm: make_env(nm, boot_wait=args.boot_wait))
    print(f"[record] booting env={args.env} ... ({len(ENV_NAMES)} envs switchable from the browser)",
          flush=True)
    SESSION = GameSession(builder, args.env, tick=args.tick)
    SESSION.reset()
    threading.Thread(target=SESSION.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)

    def _shutdown(*_):                 # graceful SIGTERM/SIGINT -> close the child game + Xvfb (no leaks)
        try: SESSION.close()
        except Exception: pass
        try: srv._BaseServer__shutdown_request = True
        except Exception: pass
        os._exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"[record] serving on http://localhost:{args.port}  (forward this port; open in your browser)",
          flush=True)
    print(f"[record] keys: {KEYHELP}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SESSION.close()
        srv.server_close()


if __name__ == "__main__":
    main()
