"""Turn ``scenes.json`` (rl.diffusion_ft.tools.dump_reach_from_select) into one
self-contained interactive page: real trajectories of failure/success scenes with
each arm's path toggling on and off, plus a time scrubber for the agents.

Cards are grouped by what the scene is EVIDENCE of -- recovered (base fails, the
bank solves), still failing (base fails and the bank does not recover it), and
preserved (the do-no-harm check) -- because a page that shows only wins is not a
result. Above them, the residual panel breaks the whole failure split down by why
each episode failed, so the still-failing cards read as instances of a measured
population rather than as anecdotes.

No model, no corpus, no GPU -- pure re-read of the bundle, so the page re-renders
whenever the scene set changes. Each scene is an SVG in world coordinates (y up):
faint lane centers, darker road edges, other-agent boxes at the current time, the
ego log dashed, the goal star, and one coloured polyline per arm. Checkboxes at
the top toggle any arm on or off; the slider sweeps agent boxes and a dot along
each path.

    python -m rl.diffusion_ft.tools.build_scene_viz --in_json scenes.json --out_html scenes.html
    python -m rl.diffusion_ft.tools.build_scene_viz ... --arms base,c_r10   # drop an arm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Colour-blind-safe; base is deliberately the muted reference.
ARM_COLORS = {"base": "#767b84", "c_r10": "#2a78d6", "c_r20": "#e8791f",
              "c_r20q": "#0a9d8e", "c_r30q": "#b25fd3", "ft": "#d64550",
              # the honest-bank A/B (submit_bank_honest.sh)
              "b_oracle": "#2a78d6", "b_cv": "#e8791f", "b_edge": "#0a9d8e"}
ARM_LABEL = {"c_r10": "c_r10 (ship, ramp10)", "c_r20": "c_r20 (comfort, ramp20)",
             "b_oracle": "bank, sees the future (oracle)",
             "b_cv": "bank, predicts cars (constant velocity)",
             "b_edge": "bank, road edges only"}
KIND_ORDER = ("recovered", "unrecovered", "regressed", "preserved")
KIND_TITLE = {
    "recovered": ("Recovered", "base violates, the bank\u2019s edit solves the scene"),
    "unrecovered": ("Still failing", "base fails and the bank does not recover it \u2014 the residual"),
    "regressed": ("Regressed", "base solves it and the bank\u2019s edit breaks it \u2014 the do-harm cases"),
    "preserved": ("Preserved", "base already solves it; the do-no-harm check"),
}
# Failure-cause composition. One episode can carry several causes at once, so the
# tags are combinations; the no-goal family shares a hue so the bar reads at a glance.
TAG_COLORS = {
    "solved": "#17a673", "no-goal": "#9b6ad4", "no-goal+overlap": "#c05fa8",
    "no-goal+offroad": "#6f5ab8", "no-goal+offroad+overlap": "#8a4f7d",
    "offroad": "#2a78d6", "overlap": "#e8791f", "offroad+overlap": "#a9571c",
    "other": "#767b84",
}
TAG_ORDER = ("solved", "no-goal", "no-goal+overlap", "no-goal+offroad",
             "no-goal+offroad+overlap", "offroad", "overlap", "offroad+overlap", "other")


def _summary_stats(select_dir, arms):
    """Read the run's select_summary.json into a compact per-arm number table.

    Returns {"repeats": int, "base": {...}, "rows": [{"arm","fail","succ","broke",
    "inter","m2","m3"}]} or None if the summary is absent. Split-level solved rates
    are the headline; m2/m3 are the corpus-marginal deltas vs base."""
    if not select_dir:
        return None
    sp = Path(select_dir) / "select_summary.json"
    if not sp.exists():
        return None
    S = json.loads(sp.read_text())
    pol = S.get("policies", {})
    # base reference: read base_solved_rate off any policy that has splits
    base = None
    for r in pol.values():
        s = r.get("splits")
        if s:
            base = {k: s[k]["base_solved_rate"] for k in
                    ("failure_heldout", "success_heldout", "intermittent_heldout") if k in s}
            break
    rows = []
    for arm in arms:
        r = pol.get(arm)
        if not r or "splits" not in r:
            continue
        s = r["splits"]
        rows.append({
            "arm": arm,
            "fail": s["failure_heldout"]["solved_rate"],
            "succ": s["success_heldout"]["solved_rate"],
            "broke": s["success_heldout"].get("broken", 0.0),
            "inter": s["intermittent_heldout"]["solved_rate"],
            "m2": r.get("marginal2", {}).get("delta"),
            "m3": r.get("marginal3", {}).get("delta"),
        })
    return {"repeats": S.get("repeats"), "base": base, "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", required=True)
    p.add_argument("--out_html", required=True)
    p.add_argument("--title", default="Deploy-time repair: failure recovery & do-no-harm")
    p.add_argument("--arms", default="",
                   help="comma list of arms to SHOW; default = every arm in the bundle. "
                        "Filters the page only -- the bundle keeps every arm, so dropping "
                        "one here is a rebuild (CPU, seconds) and never a re-dump.")
    p.add_argument("--summary_json", default="",
                   help="select_summary.json for the numbers panel; default = <select_dir>/select_summary.json")
    a = p.parse_args()

    data = json.loads(Path(a.in_json).read_text())
    arms = data["arms"]
    keep = [x for x in a.arms.split(",") if x]
    if keep:
        missing = [x for x in keep if x not in arms]
        if missing:
            raise SystemExit(f"--arms {missing} not in the bundle (has {arms})")
        arms = [x for x in arms if x in keep]
        # Drop the hidden arms from the payload too, so the page carries only what
        # it draws (paths dominate the file size) and no stale arm can leak into a
        # panel that iterates its own dict rather than ARMS.
        for sc in data["scenes"]:
            sc["paths"] = {k: v for k, v in sc["paths"].items() if k in arms}
            sc["outcomes"] = {k: v for k, v in sc["outcomes"].items() if k in arms}
        R = data.get("reasons")
        if R:
            for f in ("arms", "scene_recovery"):
                R[f] = {k: v for k, v in R.get(f, {}).items() if k in arms}
        print(f"[build_scene_viz] showing {arms} of {data['arms']}")
    colors = {arm: ARM_COLORS.get(arm, "#2a78d6") for arm in arms}
    stats = _summary_stats(str(Path(a.summary_json).parent) if a.summary_json else data.get("select_dir"), arms)
    payload = json.dumps({"arms": arms, "colors": colors,
                          "labels": {arm: ARM_LABEL.get(arm, arm) for arm in arms},
                          "stats": stats,
                          "reasons": data.get("reasons"),
                          "tag_colors": TAG_COLORS, "tag_order": TAG_ORDER,
                          "kind_order": KIND_ORDER, "kind_title": KIND_TITLE,
                          "scenes": data["scenes"]})

    html = _TEMPLATE.replace("__TITLE__", a.title).replace("__DATA__", payload)
    Path(a.out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_html).write_text(html)
    print(f"[build_scene_viz] {len(data['scenes'])} scenes -> {a.out_html}")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f5;--card:#ffffff;--ink:#141414;--muted:#5b5b57;--line:#e4e3df;--edge:#8b96a1;--lane:#dfe3e7;}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15161a;--card:#1e2026;--ink:#ececef;--muted:#9aa0a8;--line:#2c2f37;--edge:#5b6672;--lane:#33383f;}}
:root[data-theme="dark"]{--bg:#15161a;--card:#1e2026;--ink:#ececef;--muted:#9aa0a8;--line:#2c2f37;--edge:#5b6672;--lane:#33383f;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;padding:16px}
h1{font-size:19px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 14px}
.controls{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:14px;align-items:center}
.controls label{display:inline-flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
.sw{width:22px;height:4px;border-radius:2px;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.card h3{margin:0 0 2px;font-size:14px}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 6px;font-size:11px}
.badge{padding:1px 7px;border-radius:20px;border:1px solid var(--line);color:var(--muted)}
.badge.ok{color:#0a7d55;border-color:#0a7d55}
.badge.bad{color:#c0392b;border-color:#c0392b}
svg{width:100%;height:auto;display:block;background:transparent;border-radius:6px}
input[type=range]{width:220px}
.time{margin-left:auto;display:flex;align-items:center;gap:8px;color:var(--muted)}
.stats{margin:0 0 14px;overflow-x:auto}
table.nums{border-collapse:collapse;font-size:13px;min-width:520px}
table.nums th,table.nums td{padding:5px 12px;text-align:right;border-bottom:1px solid var(--line)}
table.nums th:first-child,table.nums td:first-child{text-align:left}
table.nums thead th{color:var(--muted);font-weight:600}
table.nums tr.base td{color:var(--muted)}
table.nums td.arm{font-weight:600}
.pos{color:#0a7d55}.neg{color:#c0392b}
.cap{color:var(--muted);font-size:11px;margin-top:4px}
h2.sec{font-size:15px;margin:22px 0 2px;display:flex;align-items:baseline;gap:8px}
h2.sec .n{color:var(--muted);font-weight:400;font-size:12px}
p.secsub{color:var(--muted);margin:0;font-size:12px}
.reasons{margin:16px 0 4px;border-top:1px solid var(--line);padding-top:12px}
.reasons h2{font-size:15px;margin:0 0 2px}
.rrow{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin:6px 0}
.rlab{font-size:12px;font-weight:600;text-align:right;white-space:nowrap}
.bar{display:flex;height:17px;border-radius:4px;overflow:hidden;background:var(--lane)}
.bar div{height:100%}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 2px 160px;font-size:11px;color:var(--muted)}
.legend span.k{display:inline-flex;align-items:center;gap:5px}
.legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
</style></head>
<body>
<h1>__TITLE__</h1>
<p class="sub">Base drives by default; the bank edits its own proposal only when the base is predicted to violate. Toggle any arm; scrub time to sweep the other agents.</p>
<div class="stats" id="stats"></div>
<div class="reasons" id="reasons"></div>
<div class="controls" id="controls"></div>
<div id="grid"></div>
<script>
const D = __DATA__;
const S = D.scenes, ARMS = D.arms, C = D.colors, L = D.labels;
const TC = D.tag_colors||{}, TORD = D.tag_order||[], KORD = D.kind_order||[], KT = D.kind_title||{};
const shown = {}; ARMS.forEach(a=>shown[a]=true);
let showAgents=true, showLog=true, tFrac=1.0;

function pct(x){return x==null?'—':(100*x).toFixed(1)+'%';}
function signed(x){return x==null?'—':(x>=0?'+':'')+x.toFixed(4);}
function statsPanel(){
  const el=document.getElementById('stats'); if(!D.stats){el.style.display='none';return;}
  const st=D.stats, b=st.base||{};
  let h=`<table class="nums"><thead><tr><th>arm</th>`
    +`<th>failure recovery</th><th>success non-regression</th><th>intermittent</th>`
    +`<th>corpus m2</th><th>corpus m3</th></tr></thead><tbody>`;
  st.rows.forEach(r=>{
    const brk=r.broke>0?` <span class="neg">(broke ${(100*r.broke).toFixed(1)}%)</span>`:'';
    const m2=r.m2==null?'—':`<span class="${r.m2>=0?'pos':'neg'}">${signed(r.m2)}</span>`;
    const m3=r.m3==null?'—':`<span class="${r.m3>=0?'pos':'neg'}">${signed(r.m3)}</span>`;
    h+=`<tr><td class="arm" style="color:${C[r.arm]}">${L[r.arm]||r.arm}</td>`
      +`<td>${pct(r.fail)}</td><td>${pct(r.succ)}${brk}</td><td>${pct(r.inter)}</td>`
      +`<td>${m2}</td><td>${m3}</td></tr>`;
  });
  h+=`</tbody></table><div class="cap">Solve rate per stratum vs the base line (${st.repeats}×`
    +` paired repeats/scene). m2/m3 are corpus-marginal solve-rate deltas (failures are ~1.5% of the corpus).</div>`;
  el.innerHTML=h;
}

// What the failures are MADE of, over the whole split -- the still-failing cards
// below are samples from this population, not anecdotes. The composition bar uses
// the exact cause combination per episode; the table under it counts each cause
// independently (so its row sums exceed the failure count) because that is the
// comparison that matters: the filter deletes violations, it cannot make the car
// arrive, and a flat no-goal column next to a collapsing offroad column says the
// residual is reachability, not repair strength.
function reasonsPanel(){
  const el=document.getElementById('reasons'), R=D.reasons;
  if(!R||!R.arms||!Object.keys(R.arms).length){el.style.display='none';return;}
  const armsIn=ARMS.filter(a=>R.arms[a]);
  const base=R.arms['base'];
  let h=`<h2>Why the failure split still fails</h2>`
    +`<p class="secsub">All ${R.n_scenes} <code>${R.split}</code> scenes × ${R.repeats} paired repeats `
    +`= ${base?base.n_episodes:''} episodes per arm. An episode is solved only when it reaches the goal `
    +`AND stays on-road AND avoids every agent, so a failure can carry more than one cause.</p>`;
  const seen=new Set();
  armsIn.forEach(a=>{const r=R.arms[a];
    const parts=[['solved',r.n_solved]].concat(TORD.filter(t=>t!=='solved'&&r.tags[t]).map(t=>[t,r.tags[t]]));
    Object.keys(r.tags).forEach(t=>{if(TORD.indexOf(t)<0)parts.push([t,r.tags[t]]);});
    let bar='';
    parts.forEach(([t,n])=>{ if(!n)return; seen.add(t);
      const pctw=100*n/r.n_episodes;
      bar+=`<div style="width:${pctw.toFixed(2)}%;background:${TC[t]||'#767b84'}" title="${t}: ${n} (${pctw.toFixed(1)}%)"></div>`;});
    h+=`<div class="rrow"><div class="rlab" style="color:${C[a]}">${L[a]||a}</div><div class="bar">${bar}</div></div>`;});
  h+=`<div class="legend">`+TORD.filter(t=>seen.has(t)).map(t=>
      `<span class="k"><i style="background:${TC[t]||'#767b84'}"></i>${t}</span>`).join('')+`</div>`;

  // marginal cause counts, with the delta vs base spelled out
  h+=`<table class="nums" style="margin-top:12px"><thead><tr><th>arm</th><th>failed episodes</th>`
    +`<th>no-goal</th><th>off-road</th><th>overlap</th><th>never recovered (scenes)</th></tr></thead><tbody>`;
  armsIn.forEach(a=>{const r=R.arms[a], sr=(R.scene_recovery||{})[a]||{};
    const fail=r.n_episodes-r.n_solved;
    const cell=k=>{const v=r.marginal[k];
      if(!base||a==='base')return `${v}`;
      const d=v-base.marginal[k], rel=base.marginal[k]?100*d/base.marginal[k]:0;
      const cls=d<0?'pos':(d>0?'neg':'');
      return `${v} <span class="${cls}">(${d>=0?'+':''}${d}${base.marginal[k]?`, ${rel>=0?'+':''}${rel.toFixed(0)}%`:''})</span>`;};
    const nev=sr.never==null?'—':`${sr.never} / ${R.n_scenes}`
      +(sr.never?` <span class="cap">(${sr.never_no_fire} never fired, ${sr.never_fired} fired &amp; not enough)</span>`:'');
    h+=`<tr class="${a==='base'?'base':''}"><td class="arm" style="color:${C[a]}">${L[a]||a}</td>`
      +`<td>${fail}</td><td>${cell('no_goal')}</td><td>${cell('offroad')}</td><td>${cell('overlap')}</td><td>${nev}</td></tr>`;});
  h+=`</tbody></table><div class="cap">Cause columns count each cause independently, so a row sums to more `
    +`than its failed-episode count. "never recovered" = the arm solves 0 of the ${R.repeats} repeats; `
    +`split by whether the filter ever fired in that scene.</div>`;
  el.innerHTML=h;
}

function ctrl(){
  const c = document.getElementById('controls'); c.innerHTML='';
  ARMS.forEach(a=>{
    const lab=document.createElement('label');
    lab.innerHTML=`<input type="checkbox" ${shown[a]?'checked':''}><span class="sw" style="background:${C[a]}"></span>${L[a]||a}`;
    lab.querySelector('input').onchange=e=>{shown[a]=e.target.checked;draw();};
    c.appendChild(lab);
  });
  const la=document.createElement('label');
  la.innerHTML=`<input type="checkbox" ${showAgents?'checked':''}> other agents`;
  la.querySelector('input').onchange=e=>{showAgents=e.target.checked;draw();}; c.appendChild(la);
  const lg=document.createElement('label');
  lg.innerHTML=`<input type="checkbox" ${showLog?'checked':''}> human log`;
  lg.querySelector('input').onchange=e=>{showLog=e.target.checked;draw();}; c.appendChild(lg);
  const tt=document.createElement('div'); tt.className='time';
  tt.innerHTML=`time <input type="range" min="0" max="100" value="100" id="tsl"><span id="tval">end</span>`;
  c.appendChild(tt);
  tt.querySelector('#tsl').oninput=e=>{tFrac=e.target.value/100;
    document.getElementById('tval').textContent=tFrac>=1?'end':Math.round(tFrac*100)+'%';draw();};
}

const W=340,H=300,PAD=8;
function tf(sc){ const[lx,hx,ly,hy]=sc.bbox; const sx=(W-2*PAD)/(hx-lx), sy=(H-2*PAD)/(hy-ly);
  const s=Math.min(sx,sy);
  return (x,y)=>[PAD+(x-lx)*s, H-PAD-(y-ly)*s]; }  // flip y

function poly(pts,f,attrs){ if(!pts.length)return'';
  const d=pts.map(p=>{const q=f(p[0],p[1]);return q[0].toFixed(1)+','+q[1].toFixed(1);}).join(' ');
  return `<polyline points="${d}" ${attrs}/>`; }
function dots(pts,f,r,fill){ return pts.map(p=>{const q=f(p[0],p[1]);
  return `<circle cx="${q[0].toFixed(1)}" cy="${q[1].toFixed(1)}" r="${r}" fill="${fill}"/>`;}).join(''); }
function box(cx,cy,yaw,l,w,f,stroke){ const dx=l/2,dy=w/2,cs=Math.cos(yaw),sn=Math.sin(yaw);
  const cor=[[dx,dy],[dx,-dy],[-dx,-dy],[-dx,dy]].map(([a,b])=>f(cx+a*cs-b*sn, cy+a*sn+b*cs));
  const d=cor.map(q=>q[0].toFixed(1)+','+q[1].toFixed(1)).join(' ');
  return `<polygon points="${d}" fill="none" stroke="${stroke}" stroke-width="1"/>`; }

function agentAt(ag,frac){ const t=ag.t; if(!t.length)return null;
  const k=Math.min(t.length-1, Math.round(frac*(t.length-1)));
  return {x:ag.xy[k][0], y:ag.xy[k][1], yaw:ag.yaw[k]}; }

function sceneSVG(sc){
  const f=tf(sc); let g='';
  const edge=getComputedStyle(document.body).getPropertyValue('--edge').trim();
  const lane=getComputedStyle(document.body).getPropertyValue('--lane').trim();
  g+=dots(sc.lanes,f,0.5,lane);
  g+=dots(sc.edges,f,0.9,edge);
  if(showLog) g+=poly(sc.ego_log.slice(sc.start_step),f,`fill="none" stroke="var(--muted)" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.8"`);
  if(showAgents){ sc.agents.forEach(ag=>{ const s=agentAt(ag,tFrac);
    if(s) g+=box(s.x,s.y,s.yaw,ag.l,ag.w,f,edge); }); }
  ARMS.forEach(a=>{ if(!shown[a]||!sc.paths[a])return; const p=sc.paths[a];
    g+=poly(p,f,`fill="none" stroke="${C[a]}" stroke-width="2.4" opacity="0.95"`);
    const k=Math.min(p.length-1,Math.round(tFrac*(p.length-1))); const q=f(p[k][0],p[k][1]);
    g+=`<circle cx="${q[0].toFixed(1)}" cy="${q[1].toFixed(1)}" r="3.2" fill="${C[a]}"/>`; });
  const gq=f(sc.goal[0],sc.goal[1]);
  g+=`<path d="M${gq[0]},${gq[1]-7} 2,4 -6,-3 8,0 -6,3 z" transform="translate(0,0)" fill="#0a9d55" stroke="var(--card)" stroke-width="0.8"/>`;
  g+=`<circle cx="${gq[0]}" cy="${gq[1]}" r="4.5" fill="none" stroke="#0a9d55" stroke-width="1.6"/>`;
  return `<svg viewBox="0 0 ${W} ${H}">${g}</svg>`;
}

// One card shows ONE repeat, so every badge also carries how the arm did over all
// of them (2/4): a still-failing card must not be readable as an unlucky draw.
function badges(sc){ let b=`<span class="badge">${sc.kind} · ${sc.split} · r${sc.repeat}</span>`;
  ARMS.forEach(a=>{ const o=sc.outcomes[a]; if(!o)return;
    const cls=o.solved?'ok':'bad';
    const causes=[!o.reached&&'no-goal', o.offroad&&'offroad', o.overlap&&'overlap'].filter(Boolean);
    let tag=o.solved?'solved':(causes.join('+')||'dirty');
    const sw=o.n_switch?` ·${o.n_switch}fire`:(a!=='base'&&o.fired_reps===0?' ·no fire':'');
    const rep=o.n_reps?` ${o.n_solved_reps}/${o.n_reps}`:'';
    b+=`<span class="badge ${cls}" style="border-color:${C[a]}">${a}: ${tag}${sw}${rep}</span>`; });
  return b; }

function card(sc){ const d=document.createElement('div'); d.className='card';
  d.innerHTML=`<h3>scene ${sc.scene_id}</h3><div class="badges">${badges(sc)}</div>${sceneSVG(sc)}`;
  return d; }

function draw(){ const root=document.getElementById('grid'); root.innerHTML='';
  const kinds=KORD.concat([...new Set(S.map(s=>s.kind))].filter(k=>KORD.indexOf(k)<0));
  kinds.forEach(k=>{ const list=S.filter(s=>s.kind===k); if(!list.length)return;
    const t=KT[k]||[k,''];
    const sec=document.createElement('section');
    sec.innerHTML=`<h2 class="sec">${t[0]}<span class="n">${list.length} scene${list.length>1?'s':''}</span></h2>`
      +`<p class="secsub">${t[1]||''}</p>`;
    const g=document.createElement('div'); g.className='grid';
    list.forEach(sc=>g.appendChild(card(sc)));
    sec.appendChild(g); root.appendChild(sec); }); }

statsPanel(); reasonsPanel(); ctrl(); draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
