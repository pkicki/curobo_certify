#!/usr/bin/env python3
"""Render an offline exploratory stage-timing report; never claims full K3 proof."""
import argparse
import html
import json
from pathlib import Path

TEMPLATE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Compiled K3 hull-only stage profiling</title>
<style>body{font:15px/1.5 system-ui,sans-serif;margin:20px auto;padding:0 20px;max-width:1100px;background:#f7f9fb;color:#192a42}
.warning{padding:12px 16px;background:#fff2d0;border-left:5px solid #ca871d}.panel{background:white;padding:18px;margin-top:20px;border:1px solid #d2d8e1;border-radius:8px}
label,select{font-size:16px}select{padding:8px;max-width:85vw}canvas{max-width:100%;display:block}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #ccd7e3;padding:9px;text-align:left}th{background:#e9eef5}code{font-size:12px;word-break:break-all}</style></head><body>
<h1>Compiled initial K3 Bernstein-hull profile</h1><p class="warning"><b>EXPLORATORY ONLY</b> — these are timings for a hull-only mathematical-model screen.
It does not certify endpoints, knots, strict interior speed positivity, native CUDA equivalence or complete requests.
No performance from this page is a registered Dev40/Holdout100/Confirm300 result.</p>
<div class="panel"><label for="selector">Input: </label><select id="selector"></select>
<p id="meta"></p><canvas id="stage" width="1000" height="310" role="img" aria-label="median stage costs for selected input"></canvas></div>
<div class="panel"><h2>Individual repetitions (milliseconds)</h2><table><thead><tr><th>Rep</th><th>JSON parse</th><th>Rational→interval pack</th><th>Compiled polynomial hulls</th><th>Summary</th><th>Total</th><th>Partial status</th></tr></thead><tbody id="rows"></tbody></table></div>
<script id="data" type="application/json">__DATA__</script>
<script>'use strict';const doc=JSON.parse(document.getElementById('data').textContent),runs=doc.results||[];
const selector=document.getElementById('selector'),meta=document.getElementById('meta'),rows=document.getElementById('rows'),canvas=document.getElementById('stage');
const stages=[['parse_ns','JSON parse'],['pack_ns','Interval packing'],['compiled_hull_ns','Compiled K3 hull'],['summary_ns','Summary']];
const ms=v=>Number(v)/1e6;
function draw(){const item=runs[Number(selector.value)];if(!item)return;
meta.textContent='SHA256: '+item.source_sha256+' | warmups: '+item.warmups+' | timed repeats: '+item.repeats+' | scope: '+item.scope;
let ctx=canvas.getContext('2d'),W=canvas.width;ctx.clearRect(0,0,W,canvas.height);
const ymax=Math.max(.000001,...stages.map(([key])=>ms(item.p50_ns[key])))*1.10;
let top=45,gap=58, maxWidth=630;
ctx.font='15px system-ui';ctx.fillStyle='#253957';ctx.fillText('Median stage duration (ms)',30,25);
stages.forEach(([key,label],i)=>{let y=top+i*gap,t=ms(item.p50_ns[key]);ctx.fillStyle='#253957';ctx.fillText(label,28,y+21);
ctx.fillStyle=['#3d70a0','#83aedd','#427e72','#91bdae'][i];ctx.fillRect(240,y,Math.max(1,maxWidth*t/ymax),27);
ctx.fillStyle='#253957';ctx.fillText(t.toFixed(3)+' ms',250+maxWidth*t/ymax,y+20);});
rows.replaceChildren();for(let r of item.raw){let tr=document.createElement('tr');let vals=[r.repeat,...stages.map(([key])=>ms(r[key]).toFixed(4)),ms(r.total_ns).toFixed(4),r.status];
vals.forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.appendChild(td);});rows.appendChild(tr);}}
for(let i=0;i<runs.length;i++){let el=document.createElement('option');el.value=String(i);el.textContent=runs[i].model||('Model '+i);selector.appendChild(el);}
selector.addEventListener('change',draw);if(runs.length){selector.value='0';draw();}else meta.textContent='NOT RUN: no timing inputs';</script></body></html>'''


def render(data):
    if data.get('experiment')!='COMPILED_HULL_STAGE_TIMING_NOT_FULL_CERTIFICATION':
        raise ValueError('unrecognized timing scope')
    for row in data.get('results',[]):
        if row.get('full_k3_certificate') is not False or row.get('campaign_eligible') is not False:
            raise ValueError('attempted to import a certificate or campaign output as hull-only')
    payload=json.dumps(data,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return TEMPLATE.replace('__DATA__',payload)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=render(json.loads(args.input.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(result)
    print(f'Wrote {args.out}')

if __name__=='__main__':main()
