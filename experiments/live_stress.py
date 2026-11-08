"""Live visualizer for the offline stress sim: watch one match unfold in the browser, in (scaled) real time.

It runs the same match as `stress_match.py` (real Runner, real MonitorVerifier, scripted table/station/monitor) but
paces the virtual clock against the wall clock, and serves a page that redraws ~7 times a second:

  - the relay table: hub ports, unplugged and loose cables, the tower against the 4" bar, the bin, what the
    gripper holds, how far into its current try the policy is
  - the station (state, prompt, last error), the monitor (calls, in flight), and the plan's stages
  - every table event and every harness decision (verdicts, retries, recoveries, guard), on a match timeline
  - the referee's live score (points, DQ, 8-ball)

    python experiments/live_stress.py                          # happy path at 10x, opens the browser
    python experiments/live_stress.py -s black_ball --speed 5  # any scenario from `stress_match.py --list`
    python experiments/live_stress.py --port 8765 --no-browser

Pick another scenario, seed or speed, pause, or skip to the end from the page. Speed only changes the pacing: the
match itself is the same one `stress_match.py -s NAME` scores for that seed. Logs go to runs_live/ (gitignored).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import stress_match as sm  # noqa: E402

RUNS = HERE.parent / "runs_live"
MAX_HARNESS_EVENTS = 400
QUIET = {"progress_moved", "set_prompt"}  # still logged to events.jsonl, just not worth a line in the feed


class Live:
    """Shared state between the match (main thread) and the web page (HTTP threads)."""

    def __init__(self, speed: float) -> None:
        self.lock = threading.RLock()
        self.speed = speed          # virtual seconds per wall second; 0 = as fast as possible
        self.paused = False
        self.skip = False           # run the rest of this match unpaced
        self.request: tuple[str, int] | None = None
        self.wake = threading.Event()
        self.run_id = 0
        self.ctx: dict[str, Any] | None = None
        self.hevents: list[list[Any]] = []
        self.stages: dict[str, dict[str, Any]] = {}
        self.current: str | None = None
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.started_wall = 0.0

    # ------------------------------------------------------------------ the match side
    def ask(self, scenario: str, seed: int) -> None:
        if scenario not in sm.SCENARIOS:
            raise ValueError(f"no scenario {scenario!r}")
        with self.lock:
            self.request = (scenario, int(seed))
            if self.ctx is not None and self.result is None:
                self.skip = True        # finish the current match unpaced, then start the new one
            self.paused = False
        self.wake.set()

    def observe(self, ctx: dict[str, Any]) -> None:
        clock = ctx["clock"]

        anchor = [time.monotonic(), 0.0, self.speed]  # wall time, virtual time, speed at the last re-anchor

        def pace(now: float, step: float) -> None:
            # Hold virtual time to anchor + wall/speed (so the harness's own compute doesn't slow the match down),
            # sleeping in small slices and keeping `last_real_advance` fresh: the scripted monitor advances the
            # clock itself if it looks like nobody else has for 0.2 s, which would distort the match.
            if self.paused and not self.skip:
                while self.paused and not self.skip:
                    time.sleep(0.04)
                    clock.last_real_advance = time.monotonic()
                anchor[:] = [time.monotonic(), now, self.speed]
            if self.skip or self.speed <= 0:
                anchor[:] = [time.monotonic(), now, self.speed]
            else:
                if anchor[2] != self.speed:
                    anchor[:] = [time.monotonic(), now, self.speed]
                while not self.skip and self.speed == anchor[2]:
                    delay = anchor[0] + (now - anchor[1]) / self.speed - time.monotonic()
                    if delay <= 0:
                        if delay < -1.0:   # far behind (a slow stretch): don't burst to catch up
                            anchor[:] = [time.monotonic(), now, self.speed]
                        break
                    time.sleep(min(0.04, delay))
                    clock.last_real_advance = time.monotonic()
            clock.last_real_advance = time.monotonic()

        clock.tick_hooks.append(pace)
        log = ctx["log"]
        original = log.event

        def event(name: str, **fields: Any) -> None:
            original(name, **fields)
            self._on_event(clock.now, name, fields)

        log.event = event  # type: ignore[method-assign]
        with self.lock:
            self.ctx = ctx
            self.hevents = []
            self.stages = {s.name: {"name": s.name, "status": "pending", "attempts": 0, "fails": 0}
                           for s in ctx["plan"].stages}
            self.current = None
            self.result = None
            self.error = None
            self.started_wall = time.time()

    def _on_event(self, t: float, name: str, fields: dict[str, Any]) -> None:
        with self.lock:
            st = self.stages.get(fields.get("stage", ""))
            if name == "stage_start" and st:
                st["status"] = "active"
                self.current = st["name"]
            elif name == "attempt_end" and st:
                st["attempts"] = max(st["attempts"], int(fields.get("attempt") or 0))
                if fields.get("ok"):
                    st["status"] = "done"
                else:
                    st["fails"] += 1
            elif name == "stage_already_done" and st:
                st["status"] = "done"
            elif name == "stage_skipped" and st:
                st["status"] = "skipped"
            elif name in ("guard_tripped",) and st:
                st["status"] = "guard"
            if name in QUIET:
                return
            brief = {k: v for k, v in fields.items() if k not in ("status", "frames", "question", "checkpoint")}
            self.hevents.append([round(t, 2), name, _short(brief)])
            del self.hevents[:-MAX_HARNESS_EVENTS]

    def finish(self, result: dict[str, Any] | None, error: str | None) -> None:
        with self.lock:
            self.result = result or {}
            self.error = error
            self.skip = False
            if self.current and self.stages.get(self.current, {}).get("status") == "active" and result:
                self.stages[self.current]["status"] = "failed" if not result.get("harness_ok") else "done"

    # ------------------------------------------------------------------ the page side
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            ctx = self.ctx
            out: dict[str, Any] = {
                "run_id": self.run_id, "speed": self.speed, "paused": self.paused, "skip": self.skip,
                "scenarios": [{"name": n, "note": s.note} for n, s in sm.SCENARIOS.items()],
            }
            if ctx is None:
                return out
            table, station, tr, clock, plan = ctx["table"], ctx["station"], ctx["transport"], ctx["clock"], ctx["plan"]
            s = copy.copy(table.s)
            out.update({
                "scenario": ctx["scenario"].name, "note": ctx["scenario"].note, "seed": ctx["seed"],
                "t": round(clock.now, 2), "budget": plan.time_budget_s, "block_in": table.b.block_in,
                "table": asdict(s), "policy_prompt": table.prompt, "policy_running": table.running,
                "station": {"state": station.state, "prompt": station.prompt, "arms": dict(station.arms),
                            "chunks": station.chunks, "posts": len(station.posts),
                            "error": None if not station.last_error else
                            f"{station.last_error.get('type')} {station.last_error.get('error_code') or ''}".strip()},
                "monitor": {"calls": tr.calls, "busy": tr.due is not None,
                            "outage": any(a <= clock.now < b for a, b in tr.outages)},
                "stages": list(self.stages.values()), "current": self.current,
                "harness_events": list(self.hevents),
                "table_events": [list(e) for e in table.events if not e[1].startswith("prompt")],
                "score": sm.referee(table),
                "result": self.result, "error": self.error, "log": str(ctx["log"].path),
            })
            return out


def _short(value: Any, limit: int = 140) -> Any:
    if isinstance(value, dict):
        return {k: _short(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_short(v, limit) for v in list(value)[:8]]
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 1] + "…"
    return value


# ---------------------------------------------------------------------------------------------------- HTTP
def handler(live: Live) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps(live.snapshot(), default=str).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                if path == "/api/run":
                    if "speed" in body:
                        live.speed = max(0.0, float(body["speed"]))
                    live.ask(str(body.get("scenario", "happy")), int(body.get("seed", 1)))
                elif path == "/api/control":
                    with live.lock:
                        if "speed" in body:
                            live.speed = max(0.0, float(body["speed"]))
                        if "paused" in body:
                            live.paused = bool(body["paused"])
                        if body.get("skip"):
                            live.skip = True
                else:
                    self._send(404, b"not found", "text/plain")
                    return
            except (ValueError, TypeError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            self._send(200, b'{"ok":true}', "application/json")

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--scenario", default="happy", choices=sorted(sm.SCENARIOS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", type=float, default=10.0, help="virtual seconds per wall second (0 = flat out)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    live = Live(args.speed)
    server = ThreadingHTTPServer((args.host, args.port), handler(live))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"live stress sim: {url}  (Ctrl+C to quit)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    live.ask(args.scenario, args.seed)
    RUNS.mkdir(exist_ok=True)
    try:
        while True:
            live.wake.wait()
            with live.lock:
                req, live.request = live.request, None
                live.wake.clear()
                if req is None:
                    continue
                live.run_id += 1
                live.skip = False
            name, seed = req
            print(f"match: {name} seed {seed}", flush=True)
            try:
                r = sm.run_match(sm.SCENARIOS[name], seed, out_dir=RUNS, observe=live.observe)
                summary = {k: r[k] for k in ("harness_ok", "reason", "t_end", "points", "dq", "lost", "monitor_calls")}
                live.finish(summary, None)
                print(f"  points={r['points']} lost={r['lost']} harness_ok={r['harness_ok']} ({r['reason'][:80]})",
                      flush=True)
            except Exception as exc:  # keep serving; show it on the page
                traceback.print_exc()
                live.finish(None, f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


# ---------------------------------------------------------------------------------------------------- page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Hill Climb · live stress sim</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b35;--ink:#e7e9ee;--dim:#8a93a3;--ok:#3ecf8e;--warn:#f5b84a;
--bad:#ef5a5a;--info:#5aa7ef;--violet:#a58cf5;--wood:#2a2219;--wood2:#3a2e20}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 ui-sans-serif,-apple-system,
"Segoe UI",Roboto,sans-serif}
header{display:flex;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:15px;margin:0 12px 0 0;font-weight:600}h1 span{color:var(--dim);font-weight:400}
select,input,button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;
padding:5px 8px;font:inherit}button{cursor:pointer}button:hover{border-color:#3b4250}
button.on{background:#24324a;border-color:var(--info)}button.primary{background:#1f3b2e;border-color:var(--ok)}
.grow{flex:1}.dim{color:var(--dim)}
main{display:grid;grid-template-columns:minmax(520px,1.6fr) minmax(300px,1fr);gap:12px;padding:12px 16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:0 0 8px;font-weight:600}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.pill.running{border-color:var(--ok);color:var(--ok)}.pill.starting{border-color:var(--warn);color:var(--warn)}
.pill.error{border-color:var(--bad);color:var(--bad)}.pill.idle{color:var(--dim)}
.clock{font:600 22px/1 ui-monospace,Menlo,monospace}.bar{height:6px;background:#222733;border-radius:3px;overflow:hidden}
.bar>div{height:100%;background:var(--info);transition:width .15s linear}
#timeline{position:relative;height:30px;margin-top:8px;background:#141821;border-radius:6px;border:1px solid var(--line)}
#timeline .now{position:absolute;top:0;bottom:0;width:2px;background:var(--ink);opacity:.6}
#timeline .m{position:absolute;top:5px;width:3px;height:18px;border-radius:2px}
.stages{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:6px}
.stage{padding:6px 8px;border-radius:6px;border:1px solid var(--line);background:#12151b;font-size:12px}
.stage b{display:block;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stage.active{border-color:var(--info);box-shadow:0 0 0 1px var(--info) inset}.stage.done{border-color:#21583f;background:#11211a}
.stage.skipped{opacity:.5}.stage.failed,.stage.guard{border-color:var(--bad);background:#2a1414}
.score{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.score div{background:#12151b;border-radius:8px;padding:8px}
.score b{display:block;font:600 20px/1.2 ui-monospace,Menlo,monospace}
.feed{font:12px/1.45 ui-monospace,Menlo,monospace;max-height:300px;overflow:auto;white-space:pre-wrap}
.feed div{padding:1px 0;border-bottom:1px solid #1c2029}.feed .t{color:var(--dim);display:inline-block;width:58px}
.ev-bad{color:var(--bad)}.ev-ok{color:var(--ok)}.ev-warn{color:var(--warn)}.ev-info{color:var(--info)}.ev-v{color:var(--violet)}
svg text{font-family:ui-sans-serif,-apple-system,sans-serif}
#banner{font-weight:600}
</style></head><body>
<header>
  <h1>Hill Climb <span>· live stress sim</span></h1>
  <select id="scenario"></select>
  <label class="dim">seed <input id="seed" type="number" value="1" min="1" style="width:64px"></label>
  <button class="primary" id="run">▶ Run match</button>
  <span class="dim">speed</span>
  <span id="speeds"></span>
  <button id="pause">⏸ Pause</button>
  <button id="skip">⏭ Skip to end</button>
  <span class="grow"></span><span class="dim" id="note"></span>
</header>
<main>
  <section style="display:flex;flex-direction:column;gap:12px">
    <div class="card">
      <div class="row">
        <div><div class="dim">match clock</div><div class="clock" id="clock">–</div></div>
        <div class="grow"><div class="bar"><div id="tbar" style="width:0"></div></div>
          <div id="timeline"></div></div>
      </div>
    </div>
    <div class="card"><h2>Relay table</h2><svg id="scene" viewBox="0 0 640 330" width="100%"></svg>
      <div class="row" style="margin-top:6px"><span class="dim">policy prompt</span><span id="pprompt" class="grow">–</span></div>
      <div class="row"><span class="dim">current try</span><div class="bar grow"><div id="phase" style="background:var(--violet)"></div></div></div>
    </div>
    <div class="card"><h2>Plan stages</h2><div class="stages" id="stages"></div></div>
  </section>
  <section style="display:flex;flex-direction:column;gap:12px">
    <div class="card"><h2>Referee (rubric v4)</h2>
      <div class="score"><div><span class="dim">points</span><b id="pts">0</b></div>
        <div><span class="dim">tower</span><b id="tower">–</b></div><div><span class="dim">balls</span><b id="balls">0</b></div></div>
      <div id="banner" style="margin-top:8px"></div>
    </div>
    <div class="card"><h2>Station · monitor</h2>
      <div class="row"><span class="pill" id="sstate">–</span><span class="dim">chunks</span><span id="chunks">0</span>
        <span class="dim">POSTs</span><span id="posts">0</span><span class="dim">monitor calls</span><span id="mcalls">0</span>
        <span class="pill" id="mbusy">monitor idle</span></div>
      <div style="margin-top:6px" class="ev-bad" id="serr"></div>
    </div>
    <div class="card"><h2>Table events</h2><div class="feed" id="tfeed"></div></div>
    <div class="card"><h2>Harness decisions</h2><div class="feed" id="hfeed" style="max-height:380px"></div></div>
  </section>
</main>
<script>
const $=id=>document.getElementById(id);
const SPEEDS=[1,2,5,10,30,0];let state={},picked=false;
$('speeds').innerHTML=SPEEDS.map(s=>`<button data-s="${s}">${s?s+'×':'max'}</button>`).join(' ');
document.querySelectorAll('#speeds button').forEach(b=>b.onclick=()=>post('/api/control',{speed:+b.dataset.s}));
$('run').onclick=()=>post('/api/run',{scenario:$('scenario').value,seed:+$('seed').value});
$('pause').onclick=()=>post('/api/control',{paused:!state.paused});
$('skip').onclick=()=>post('/api/control',{skip:true});
function post(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(tick)}
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function evClass(e){e=e.toLowerCase();
  if(/dq|black ball in|rule:|error|guard_tripped|failed|knocked|arm/.test(e))return'ev-bad';
  if(/unplugged|plugged in|tower \d|ball in bin|ok": true|done|run_end/.test(e))return'ev-ok';
  if(/retry|recovery|refused|restart|timeout|regression|blind|no_frames|same_frames|watch_lost|black ball/.test(e))return'ev-warn';
  if(/stop|start/.test(e))return'ev-info';return''}
function markColor(e){const c=evClass(e);return c=='ev-bad'?'var(--bad)':c=='ev-ok'?'var(--ok)':c=='ev-warn'?'var(--warn)':'var(--info)'}
function fmtFields(f){return Object.entries(f||{}).map(([k,v])=>k+'='+(typeof v=='object'?JSON.stringify(v):v)).join(' ')}

function scene(s){
  const T=s.table,held=T.held,bi=s.block_in||1.5;let g='';
  g+=`<rect x="10" y="40" width="620" height="280" rx="14" fill="var(--wood)" stroke="var(--wood2)"/>`;
  // hub: 4 original ports + 4 free ports
  g+=`<rect x="40" y="150" width="150" height="86" rx="8" fill="#1d2230" stroke="#394158"/>
      <text x="115" y="250" fill="var(--dim)" font-size="11" text-anchor="middle">USB hub</text>`;
  for(let i=0;i<4;i++){const x=52+i*34;
    const orig=i<Math.min(T.hub-T.replugged,4),rep=!orig&&i<T.hub;
    g+=`<rect x="${x}" y="160" width="24" height="16" rx="3" fill="${orig?'var(--ok)':rep?'var(--bad)':'#0c0e13'}" stroke="#4a536a"/>`;
    const p=i<T.plugged;
    g+=`<rect x="${x}" y="200" width="24" height="16" rx="3" fill="${p?'var(--info)':'#0c0e13'}" stroke="#4a536a"/>`;}
  g+=`<text x="44" y="194" fill="var(--dim)" font-size="10">start ports</text><text x="44" y="232" fill="var(--dim)" font-size="10">free ports</text>`;
  // unplugged cables lying on the table, and loose ones
  const lying=Math.max(0,T.out-T.replugged-(held=='cable_out'?1:0));
  for(let i=0;i<lying;i++){const y=258+i*12;g+=`<path d="M40 ${y} q20 -10 40 0 t40 0" fill="none" stroke="var(--ok)" stroke-width="3" opacity=".8"/><rect x="118" y="${y-4}" width="10" height="8" fill="#cfd5e2"/>`}
  for(let i=0;i<T.loose;i++){const y=70+i*14;g+=`<path d="M40 ${y} q25 10 50 0 t50 0" fill="none" stroke="var(--info)" stroke-width="3" opacity=".75"/><rect x="138" y="${y-4}" width="10" height="8" fill="#cfd5e2"/>`}
  g+=`<text x="40" y="58" fill="var(--dim)" font-size="11">loose cables (${T.loose})</text>`;
  // tower vs 4" bar
  const px=18,base=300,bx=300;
  g+=`<line x1="${bx-50}" x2="${bx+70}" y1="${base-4*px}" y2="${base-4*px}" stroke="var(--warn)" stroke-dasharray="5 4"/>
      <text x="${bx+74}" y="${base-4*px+4}" fill="var(--warn)" font-size="11">4"</text>`;
  if(T.max_tower>T.tower)g+=`<rect x="${bx-20}" y="${base-T.max_tower*bi*px}" width="40" height="${(T.max_tower-T.tower)*bi*px}" fill="none" stroke="#5b6475" stroke-dasharray="3 3"/>`;
  const cols=['#e86b5a','#f0b04a','#5ac08f','#5aa7ef','#a58cf5','#e86bc0','#9bd35a'];
  for(let i=0;i<T.tower;i++){const h=bi*px;g+=`<rect x="${bx-20}" y="${base-(i+1)*h}" width="40" height="${h-2}" rx="3" fill="${cols[i%cols.length]}"/>`}
  g+=`<text x="${bx}" y="318" fill="var(--dim)" font-size="11" text-anchor="middle">tower ${T.tower} · ${(T.tower*bi).toFixed(1)}"</text>`;
  // bin
  g+=`<path d="M480 200 L600 200 L590 300 L490 300 Z" fill="#141821" stroke="#4a536a"/>
      <text x="540" y="318" fill="var(--dim)" font-size="11" text-anchor="middle">bin</text>`;
  const balls=[...Array(T.balls).fill('#f2f2f2'),...Array(T.black).fill('#050505')];
  balls.forEach((c,i)=>{const x=502+(i%5)*20,y=286-Math.floor(i/5)*20;g+=`<circle cx="${x}" cy="${y}" r="8" fill="${c}" stroke="${c=='#050505'?'var(--bad)':'#999'}" stroke-width="${c=='#050505'?2:1}"/>`});
  // grippers (the sim's held object is in the right one)
  const gy=T.arms_up?60:140;
  g+=gripper(240,T.arms_up?60:120,null,'left')+gripper(400,gy,held,'right');
  return g;
}
function gripper(x,y,held,label){
  const open=!held;let g=`<line x1="${x}" y1="0" x2="${x}" y2="${y}" stroke="#6b7486" stroke-width="6"/>
    <rect x="${x-22}" y="${y}" width="44" height="10" rx="3" fill="#8a93a3"/>
    <rect x="${x-(open?24:14)}" y="${y+10}" width="6" height="24" fill="#8a93a3"/><rect x="${x+(open?18:8)}" y="${y+10}" width="6" height="24" fill="#8a93a3"/>
    <text x="${x+30}" y="${y+8}" fill="var(--dim)" font-size="10">${label}</text>`;
  if(held=='cable_out'||held=='cable_loose')g+=`<rect x="${x-6}" y="${y+16}" width="12" height="14" fill="#cfd5e2"/><path d="M${x} ${y+30} q-10 25 10 40" stroke="var(--ok)" stroke-width="3" fill="none"/>`;
  if(held=='block')g+=`<rect x="${x-9}" y="${y+14}" width="18" height="18" rx="2" fill="#f0b04a"/>`;
  if(held=='ball')g+=`<circle cx="${x}" cy="${y+24}" r="8" fill="#f2f2f2"/>`;
  if(held=='black')g+=`<circle cx="${x}" cy="${y+24}" r="8" fill="#050505" stroke="var(--bad)" stroke-width="2"/><text x="${x}" y="${y+28}" fill="#fff" font-size="8" text-anchor="middle">8</text>`;
  return g;
}

function render(s){
  if(s.scenarios&&!$('scenario').options.length){
    $('scenario').innerHTML=s.scenarios.map(x=>`<option value="${x.name}" title="${esc(x.note)}">${x.name}</option>`).join('')}
  if(s.scenario&&!picked){$('scenario').value=s.scenario;$('seed').value=s.seed;picked=true}
  document.querySelectorAll('#speeds button').forEach(b=>b.classList.toggle('on',+b.dataset.s===s.speed));
  $('pause').textContent=s.paused?'▶ Resume':'⏸ Pause';$('pause').classList.toggle('on',s.paused);
  if(!s.table)return;
  $('note').textContent=`${s.scenario} · seed ${s.seed}${s.note?' — '+s.note:''}`;
  $('clock').textContent=`${s.t.toFixed(1)} s / ${s.budget|0} s`;
  $('tbar').style.width=Math.min(100,100*s.t/s.budget)+'%';
  $('scene').innerHTML=scene(s);
  $('pprompt').textContent=(s.policy_running?'▶ ':'■ ')+(s.policy_prompt||'–');
  $('phase').style.width=(s.policy_running?100*s.table.phase:0)+'%';
  $('stages').innerHTML=s.stages.map(st=>`<div class="stage ${st.status}"><b>${st.name}</b>
    <span class="dim">${st.status}${st.attempts?` · try ${st.attempts}`:''}${st.fails?` · ${st.fails} failed`:''}</span></div>`).join('');
  const sc=s.score;$('pts').textContent=sc.points;$('tower').textContent=(s.table.tower*s.block_in).toFixed(1)+'"';
  $('balls').textContent=sc.balls+(sc.black_in_bin?` +${sc.black_in_bin}⚫`:'');
  let ban='';if(sc.dq.length)ban=`<span class="ev-bad">DQ — ${esc(sc.dq[0])}</span>`;
  else if(sc.black_in_bin)ban='<span class="ev-bad">LOST — 8-ball in the bin</span>';
  if(s.result){ban+=(ban?'<br>':'')+`<span class="${s.result.harness_ok?'ev-ok':'ev-warn'}">match over at ${s.result.t_end}s · harness ${s.result.harness_ok?'ok':'stopped'}: ${esc(s.result.reason)}</span>`}
  else if(s.skip)ban+=(ban?'<br>':'')+'<span class="dim">skipping to the end…</span>';
  if(s.error)ban+=`<br><span class="ev-bad">${esc(s.error)}</span>`;
  $('banner').innerHTML=ban;
  const st=s.station;$('sstate').textContent=st.state;$('sstate').className='pill '+st.state;
  $('chunks').textContent=st.chunks;$('posts').textContent=st.posts;$('mcalls').textContent=s.monitor.calls;
  $('mbusy').textContent=s.monitor.outage?'API outage':s.monitor.busy?'monitor thinking…':'monitor idle';
  $('mbusy').className='pill '+(s.monitor.outage?'error':s.monitor.busy?'starting':'idle');
  $('serr').textContent=st.error?('last error: '+st.error):'';
  feed('tfeed',s.table_events.map(([t,e])=>[t,e,e]));
  feed('hfeed',s.harness_events.map(([t,n,f])=>[t,`<b>${n}</b> ${esc(fmtFields(f))}`,n+' '+JSON.stringify(f)]),true);
  const tl=$('timeline');tl.innerHTML=s.table_events.concat(s.harness_events.filter(e=>/retry|guard_tripped|station_restart|recovery|stop/.test(e[1])).map(e=>[e[0],e[1]]))
    .map(([t,e])=>`<div class="m" title="${t}s ${esc(e)}" style="left:${100*t/s.budget}%;background:${markColor(e)}"></div>`).join('')
    +`<div class="now" style="left:${Math.min(100,100*s.t/s.budget)}%"></div>`;
}
function feed(id,rows,html){const el=$(id),stick=el.scrollTop+el.clientHeight>=el.scrollHeight-20;
  el.innerHTML=rows.map(([t,txt,raw])=>`<div class="${evClass(raw)}"><span class="t">${(+t).toFixed(1)}</span>${html?txt:esc(txt)}</div>`).join('');
  if(stick)el.scrollTop=el.scrollHeight}
let busy=false;function tick(){if(busy)return;busy=true;fetch('/api/state').then(r=>r.json()).then(s=>{state=s;render(s)})
  .catch(()=>{}).finally(()=>busy=false)}
setInterval(tick,150);tick();
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
