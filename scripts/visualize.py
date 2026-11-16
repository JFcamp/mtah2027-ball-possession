"""Local replay of the tracking with the pipeline's labels: a self-contained HTML file.

    python scripts/visualize.py [--start 11185] [--seconds 90] [--version v4] [--out viz/replay.html]

Shows the pitch, the players of both teams, the ball (cleaned track), the label the pipeline gives to
every frame (H / A / D) and, if the window overlaps the public ground truth, that label too. The video
itself (.mp4) is not part of the data we have, so this replays the tracking, not the footage.

The output holds frame-level tracking positions of the licensed match data, so it is written to
viz/ (git-ignored) and must never be committed or shared.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import io as sio  # noqa: E402
from src import run as runner  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
STEP = 2  # every 2nd frame (12.5 fps) keeps the file small
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>branca analytics: replay</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;background:#12171d;color:#eee;margin:0;padding:12px}
#wrap{max-width:1100px;margin:auto} canvas{width:100%;background:#2e7d32;border-radius:6px}
#bar{display:flex;gap:10px;align-items:center;margin:8px 0;flex-wrap:wrap}
.tag{padding:4px 12px;border-radius:14px;font-weight:600;min-width:120px;text-align:center}
button,select{background:#2b3440;color:#eee;border:1px solid #556;border-radius:4px;padding:4px 10px}
input[type=range]{flex:1;min-width:240px} small{color:#9ab}
#strip{width:100%;height:26px;display:block;border-radius:4px;margin-top:4px}
</style></head><body><div id="wrap">
<h3 style="margin:4px 0">branca analytics: replay of the tracking with the pipeline's labels</h3>
<canvas id="c" width="1050" height="680"></canvas>
<div id="bar"><button id="play">Pause</button>
<select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option>
<option value="2">2x</option><option value="4">4x</option></select>
<input id="seek" type="range" min="0" value="0"><span id="clock"></span></div>
<div id="bar2" style="display:flex;gap:10px;align-items:center">
<span>pipeline:</span><span class="tag" id="pred"></span><span id="gtwrap">reference (public GT):</span>
<span class="tag" id="gt"></span><small id="info"></small></div>
<canvas id="strip" height="26"></canvas>
<small>Top strip: pipeline labels over the window (blue = home H, red = away A, grey = dead ball D).
Second strip, if present: public ground truth. Ball position is the cleaned track; no ball is drawn
where the tracker lost it.</small></div>
<script>
const D=__DATA__;
const L=105.3,W=68,cv=document.getElementById('c'),g=cv.getContext('2d');
const COL=['#1e88e5','#e53935','#9e9e9e','#444'],NAME=['H  home possession','A  away possession','D  dead ball','X'];
const sx=x=>20+x/L*(cv.width-40), sy=y=>20+y/W*(cv.height-40);
function pitch(){g.fillStyle='#2e7d32';g.fillRect(0,0,cv.width,cv.height);g.strokeStyle='#e8f5e9';g.lineWidth=2;
 g.strokeRect(sx(0),sy(0),sx(L)-sx(0),sy(W)-sy(0));g.beginPath();g.moveTo(sx(L/2),sy(0));g.lineTo(sx(L/2),sy(W));g.stroke();
 g.beginPath();g.arc(sx(L/2),sy(W/2),9.15/L*(cv.width-40),0,7);g.stroke();
 for(const s of [0,1]){const x0=s?L:0,dir=s?-1:1;g.strokeRect(sx(s?L-16.5:0),sy(W/2-20.16),16.5/L*(cv.width-40),40.32/W*(cv.height-40));
 g.strokeRect(sx(s?L-5.5:0),sy(W/2-9.16),5.5/L*(cv.width-40),18.32/W*(cv.height-40));}}
function strip(id,arr,y){const s=document.getElementById(id);}
const sc=document.getElementById('strip'),sg=sc.getContext('2d');
function drawStrip(){if(sc.width!==sc.clientWidth)sc.width=sc.clientWidth;const n=D.frames.length,rows=D.hasGt?2:1;sc.height=14*rows+2;
 for(let i=0;i<n;i++){const x=i/n*sc.width,w=Math.max(1,sc.width/n+0.5);
  sg.fillStyle=COL[D.frames[i].p];sg.fillRect(x,0,w,13);
  if(D.hasGt){sg.fillStyle=COL[D.frames[i].g<0?3:D.frames[i].g];sg.fillRect(x,15,w,13);}}}
const seek=document.getElementById('seek');seek.max=D.frames.length-1;
let i=0,playing=true,last=performance.now(),acc=0;
function tag(el,c){el.textContent=NAME[c];el.style.background=COL[c];}
function draw(){pitch();const f=D.frames[i];
 for(const p of f.pl){g.beginPath();g.fillStyle=p[2]?'#e53935':'#1e88e5';g.arc(sx(p[0]),sy(p[1]),7,0,7);g.fill();g.strokeStyle='#fff';g.lineWidth=1.5;g.stroke();}
 if(f.b){g.beginPath();g.fillStyle='#fff';g.arc(sx(f.b[0]),sy(f.b[1]),6,0,7);g.fill();g.strokeStyle='#000';g.lineWidth=2;g.stroke();}
 tag(document.getElementById('pred'),f.p);
 const gt=document.getElementById('gt');if(D.hasGt){tag(gt,f.g<0?3:f.g);if(f.g<0)gt.textContent='-'}else{gt.style.display='none';document.getElementById('gtwrap').style.display='none'}
 const sec=(f.t/25);document.getElementById('clock').textContent='frame '+f.t+'  |  '+Math.floor(sec/60)+':'+String(Math.floor(sec%60)).padStart(2,'0');
 document.getElementById('info').textContent=f.b?'ball seen':'ball not tracked';
 seek.value=i;
 drawStrip._x=i;}
function tick(now){const dt=now-last;last=now;if(playing){acc+=dt*parseFloat(document.getElementById('speed').value);
 const per=1000/(25/D.step);while(acc>=per){acc-=per;i=(i+1)%D.frames.length;}}
 draw();
 drawStrip();sg.strokeStyle='#fff';sg.lineWidth=2;const x=i/D.frames.length*sc.width;sg.beginPath();sg.moveTo(x,0);sg.lineTo(x,sc.height);sg.stroke();
 requestAnimationFrame(tick);}
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play';};
seek.oninput=()=>{i=parseInt(seek.value);};
window.onresize=drawStrip;drawStrip();requestAnimationFrame(tick);
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=11185, help="first video frame")
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--version", default="v4")
    parser.add_argument("--out", default=os.path.join(ROOT, "viz", "replay.html"))
    args = parser.parse_args()

    result = runner.build(args.version)
    tracking = sio.load_tracking()
    stop = min(args.start + args.seconds * sio.FPS, sio.N_FRAMES)
    gt = None
    if os.path.isfile(GT_PATH):
        gt = np.array([CODE[v] for v in pd.read_csv(GT_PATH)["label"]], dtype=np.int8)

    frames = []
    has_gt = False
    for t in range(args.start, stop, STEP):
        players = []
        for k in range(tracking.players_xy.shape[1]):
            x, y = tracking.players_xy[t, k]
            if np.isfinite(x):
                players.append([round(float(x), 1), round(float(y), 1), int(tracking.team_of_slot[k])])
        bx, by = result.track.xy[t]
        label = int(result.labels[t])
        g = int(gt[t]) if gt is not None and gt[t] < 3 else -1
        has_gt |= g >= 0
        frames.append({"t": t, "pl": players,
                       "b": [round(float(bx), 1), round(float(by), 1)] if np.isfinite(bx) else None,
                       "p": label if label < 3 else 3, "g": g})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = {"frames": frames, "step": STEP, "hasGt": has_gt}
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print("wrote", os.path.relpath(args.out, ROOT), "{} frames shown, {:.0f} KB".format(
        len(frames), os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    main()
