/* Interactive offline proof explorer. This UI never changes or certifies a
 * native cuRobo trajectory. Floating point is used for PLOTS ONLY. */
(function () {
'use strict';
const P=window.BernsteinProof;
const $=id=>document.getElementById(id);
const defaults=[
 {id:'false-alarm',title:'Feasible · misleading convex hull',tag:'The essential example',family:'generic',coefficients:['-1','2/5','-1'],
  description:'The middle Bernstein coefficient is positive, but the whole quadratic is below zero. One midpoint split makes both child hulls nonpositive.'},
 {id:'violation',title:'Actual violation · positive witness',tag:'Infeasible',family:'generic',coefficients:['-1','2','-1'],
  description:'The polynomial itself is positive at s=1/2. The exact midpoint evaluation is a rigorous violation witness—not merely an uncertain hull.'},
 {id:'tangent',title:'Tangency · strict squared-speed positivity',tag:'Strict versus nonstrict',family:'positive_x',strict:true,coefficients:['-1/4','1/4','-1/4'],
  description:'r(s)=−(s−1/2)² touches zero inside the interval. It satisfies r≤0 but violates r<0 for interior squared speed. Turn strict checking off to see the nonstrict certificate.'},
 {id:'degree9',title:'Ninth-degree jerk-like residual',tag:'Higher-degree hull',family:'jerk',coefficients:['-1/5','-1/5','-1/5','-1/5','1/10','1/10','-1/5','-1/5','-1/5','-1/5'],
  description:'A deliberately synthetic degree-nine residual, similar in degree to a squared-jerk inequality. The positive coefficients do not imply a violation.'},
 {id:'safe',title:'Already proven · no splitting',tag:'Fast path',family:'velocity',coefficients:['-1','-1/2','-3/4','-1/3'],
  description:'Every initial Bernstein coefficient is nonpositive, so the original interval itself is a complete proof leaf.'},
 {id:'zero',title:'Identically zero residual',tag:'Boundary case',family:'generic',coefficients:['0','0','0'],
  description:'The nonstrict inequality is true everywhere. Under a strict inequality it fails everywhere: zero is not negative.'}
];
let library=defaults.slice();
let current=null,proof=null, cursor=0,focus=-1, playback=null;
const familyNames={generic:'Generic r(s) ≤ 0',positive_x:'Squared speed: −x(s) ≤ 0',velocity:'Velocity',acceleration_plus:'Acceleration (+)',acceleration_minus:'Acceleration (−)',jerk:'Jerk²'};
const stateTag={PROVEN:'Proven',SPLIT:'Split',VIOLATED:'Violation',UNKNOWN_DEPTH:'Depth limit',UNKNOWN_NODES:'Node limit',PENDING:'Pending'};
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(v,digits=4) {
 const x=P.num(v);
 return Number.isFinite(x)?x.toFixed(digits).replace(/\.?0+$/,''):P.repr(v);
}
function statusKind(status){return status==='CERTIFIED'?'good':status.startsWith('VIOLATED')?'bad':'warn';}
function stop(){if(playback){clearInterval(playback);playback=null;}$('play').textContent='▶ Play';}
function selectedExample(){
 const id=$('preset').value;
 return library.find(x=>x.id===id) || library[0];
}
function loadExample(example) {
 stop();current=example;
 $('coeffs').value=example.coefficients.join(', ');
 $('strict').checked=!!example.strict;
 $('depth').value='8'; $('nodes').value='2048';
 $('exampleTitle').textContent=example.title;
 $('description').textContent=example.description||'Imported exact rational residual coefficients.';
 $('family').textContent=familyNames[example.family]||example.family||familyNames.generic;
 $('preset').value=example.id;
 $('validation').textContent='';
 run();
}
function loadSelect() {
 $('preset').innerHTML=library.map(e=>'<option value="'+escapeHtml(e.id)+'">'+escapeHtml(e.title)+'</option>').join('');
}
function settings(){
 const raw=$('coeffs').value.split(/[,\n;]/).map(x=>x.trim()).filter(Boolean);
 return {coefficients:P.parseCoeffs(raw),depth:Number($('depth').value),nodes:Number($('nodes').value),strict:$('strict').checked};
}
function run() {
 stop();
 try{
  const s=settings();
  proof=P.build(s.coefficients,s);
  cursor=proof.nodes.length;focus=proof.nodes.length-1;
  $('validation').textContent='';
  $('validation').className='';
  render();
 }catch(e) {
  proof=null;$('validation').textContent='Input error: '+e.message;
  $('validation').className='alert';
  $('verdict').textContent='INVALID INPUT';$('verdict').className='pill bad';
  $('plot').innerHTML='';$('tree').innerHTML='';$('inspector').textContent='Correct the input to construct a proof.';
 }
}
function shown(){return proof?proof.nodes.slice(0,cursor):[];}
function metric(target,value){$(target).textContent=value;}
function renderMetrics(){
 if(!proof)return;
 const finished=cursor===proof.nodes.length, status=finished?proof.status:'IN PROGRESS';
 $('verdict').textContent=status.replaceAll('_',' ');
 $('verdict').className='pill '+(finished?statusKind(status):'warn');
 const B=proof.original.map(P.parse);
 metric('degree',String(proof.degree));
 metric('initialHull',P.repr(P.max?P.max(B):B.reduce((x,y)=>P.cmp(x,y)>0?x:y)));
 metric('leaves',String(shown().filter(n=>n.state==='PROVEN').length));
 metric('processed',String(shown().filter(n=>n.state!=='PENDING').length));
 const ok=finished&&proof.replay.pass;
 $('replay').textContent=finished? (ok?'PASS · shared-arithmetic replay':'Not a complete certificate'):'Not yet complete';
 $('replay').className=ok?'mini-good':'mini-muted';
 $('stepCounter').textContent=cursor+' / '+proof.nodes.length+' nodes';
 $('prev').disabled=cursor===0;$('next').disabled=cursor===proof.nodes.length;
}
function renderTree(){
 if(!proof)return;
 const elements=proof.nodes.map((n,i)=>{
  if(i>=cursor)return '<button class="node unseen" disabled><b>'+escapeHtml(n.path||'root')+'</b><small>unseen</small></button>';
  const kind=n.state==='PROVEN'?'good':n.state==='VIOLATED'?'bad':n.state==='SPLIT'?'neutral':'warn';
  return '<button class="node '+kind+(focus===i?' focused':'')+'" data-index="'+i+'"><b>'+
   escapeHtml(n.path||'root')+'</b><small>'+escapeHtml(stateTag[n.state]||n.state)+'</small></button>';
 });
 $('tree').innerHTML=elements.join('');
 $('tree').querySelectorAll('[data-index]').forEach(btn=>btn.addEventListener('click',()=>{
  focus=Number(btn.dataset.index); render();
 }));
}
function plot(){
 if(!proof)return;
 const b=proof.original.map(P.parse), samples=240;
 const nodes=shown().filter(n=>n.state==='PROVEN'||n.state==='VIOLATED'||n.state==='UNKNOWN_DEPTH'||n.state==='PENDING');
 const active=proof.nodes[Math.min(focus,cursor-1)]||null;
 const values=Array.from({length:samples+1},(_,i)=>P.sample(b,i/samples));
 const hullVals=[0,...values];
 for(const n of nodes)hullVals.push(P.num(n.lower),P.num(n.upper));
 if(active)hullVals.push(P.num(active.lower),P.num(active.upper));
 const finite=hullVals.filter(Number.isFinite);
 if(finite.length!==hullVals.length){
  $('plot').innerHTML='<text x="30" y="35" fill="#f2ba7e">Cannot plot values outside finite JavaScript Number range; exact proof remains available.</text>';return;
 }
 const W=1000,H=400,left=64,right=24,top=18,bottom=51;
 const mn=Math.min(...finite),mx=Math.max(...finite),span=Math.max(0.3,mx-mn);
 const min=mn-0.12*span,max=mx+0.12*span;
 const X=s=>left+s*(W-left-right);
 const Y=v=>top+(max-v)/(max-min)*(H-top-bottom);
 let out='';
 for(let i=0;i<=5;i++){
  const y=min+(max-min)*i/5;
  out+='<line x1="'+left+'" y1="'+Y(y)+'" x2="'+(W-right)+'" y2="'+Y(y)+'" class="gridline"/>';
  out+='<text x="52" y="'+(Y(y)+4)+'" class="tick" text-anchor="end">'+y.toFixed(2)+'</text>';
 }
 for(let i=0;i<=10;i++){
  const x=X(i/10);
  out+='<line x1="'+x+'" y1="'+top+'" x2="'+x+'" y2="'+(H-bottom)+'" class="gridline vertical"/>';
  if(i%2===0)out+='<text x="'+x+'" y="'+(H-25)+'" class="tick" text-anchor="middle">'+(i/10).toFixed(1)+'</text>';
 }
 if(min<=0&&max>=0)out+='<line x1="'+left+'" y1="'+Y(0)+'" x2="'+(W-right)+'" y2="'+Y(0)+'" class="zero"/>';
 for(const n of nodes) {
  const l=P.num(n.l),r=P.num(n.r),lo=P.num(n.lower),hi=P.num(n.upper);
  const cls=n.state==='PROVEN'?'hull-good':n.state==='VIOLATED'?'hull-bad':'hull-unknown';
  out+='<rect class="'+cls+'" x="'+X(l)+'" y="'+Y(hi)+'" width="'+Math.max(1,X(r)-X(l))+
        '" height="'+Math.max(1,Y(lo)-Y(hi))+'"><title>'+escapeHtml(n.path||'root')+
        ': exact Bernstein hull ['+escapeHtml(P.repr(n.lower))+', '+escapeHtml(P.repr(n.upper))+']</title></rect>';
 }
 const pts=values.map((v,i)=>X(i/samples)+','+Y(v)).join(' ');
 out+='<polyline class="curve" points="'+pts+'"/>';
 if($('polygon').checked && active){
  const local=active.b.map(P.num),l=P.num(active.l),r=P.num(active.r),n=local.length-1;
  out+='<polyline class="control" points="'+local.map((v,i)=>X(l+(r-l)*i/n)+','+Y(v)).join(' ')+'"/>';
  local.forEach((v,i)=>out+='<circle class="control-dot" cx="'+X(l+(r-l)*i/n)+'" cy="'+Y(v)+'" r="4.5"/>');
 }
 const t=Number($('probe').value)/1000;
 const probe=P.at(b,P.R(BigInt($('probe').value),1000n));
 if(Number.isFinite(P.num(probe))){
  out+='<line class="probe-line" x1="'+X(t)+'" y1="'+top+'" x2="'+X(t)+'" y2="'+(H-bottom)+'"/>';
  out+='<circle class="probe-dot" cx="'+X(t)+'" cy="'+Y(P.num(probe))+'" r="5.5"/>';
 }
 if(proof.witness&&cursor===proof.nodes.length){
  const w=proof.witness, sx=P.num(w.at),sy=P.num(w.value);
  out+='<circle class="witness-dot" cx="'+X(sx)+'" cy="'+Y(sy)+'" r="10"/>';
  out+='<text class="witness-label" x="'+Math.min(W-190,X(sx)+12)+'" y="'+(Y(sy)-12)+'">Exact '+escapeHtml(w.kind)+' witness</text>';
 }
 out+='<text class="axis-name" x="'+(W/2)+'" y="'+(H-5)+'" text-anchor="middle">Path parameter s ∈ [0,1]</text>';
 $('plot').innerHTML=out;
 metric('probeValue','s='+t.toFixed(3)+' → r(s) = '+P.repr(probe)+' ≈ '+fmt(probe,6));
}
function inspect(){
 if(!proof)return;
 const n=proof.nodes[Math.min(focus,cursor-1)]||null;
 if(!n){$('inspector').innerHTML='<p>Advance one node to inspect its interval and exact bounds.</p>';return;}
 let explanation='';
 if(n.state==='PROVEN')explanation='All Bernstein coefficients are ≤ 0 on this child interval. The convex-hull property proves the inequality at every point—not just at sampled points.';
 else if(n.state==='SPLIT')explanation='At least one Bernstein coefficient is positive, but this is <b>not</b> evidence of a trajectory violation. Midpoint de Casteljau subdivision preserves the polynomial exactly and tightens local hulls.';
 else if(n.state==='VIOLATED')explanation='The polynomial itself has an exact noncompliant point (endpoint or midpoint). This is a rigorous witness; it is not inferred from a positive control coefficient.';
 else explanation='The configured proof budget was exhausted. Inconclusive does not mean violated.';
 let html='<p class="inspector-lede">'+explanation+'</p>';
 html+='<div class="inspect-grid"><div><span>Current dyadic interval</span><strong>['+P.repr(n.l)+', '+P.repr(n.r)+']</strong></div>';
 html+='<div><span>Exact coefficient bounds</span><strong>['+P.repr(n.lower)+', '+P.repr(n.upper)+']</strong></div>';
 html+='<div><span>Decision</span><strong>'+escapeHtml(stateTag[n.state]||n.state)+'</strong></div>';
 html+='<div><span>Subdivision depth</span><strong>'+n.level+'</strong></div></div>';
 html+='<div class="coef-row">'+n.b.map((v,i)=>'<div class="coef-pill"><small>b'+i+'</small><b>'+escapeHtml(P.repr(v))+'</b></div>').join('')+'</div>';
 if(proof.witness&&n.state==='VIOLATED')html+='<p class="witness-text">Exact witness: s = <b>'+P.repr(proof.witness.at)+
  '</b>, r(s) = <b>'+P.repr(proof.witness.value)+'</b>. '+(proof.witness.kind==='ZERO'?'Strict inequality requires r(s) < 0 at every interior point.':'A positive residual proves the nonstrict inequality fails.')+'</p>';
 $('inspector').innerHTML=html;
}
function render(){
 if(!proof)return;
 if(cursor>0 && (focus>=cursor || focus<0)) focus=cursor-1;
 renderMetrics();renderTree();plot();inspect();
}
function download(name,body){
 const blob=new Blob([body],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),500);
}
function fileImport(file){
 if(!file)return;
 file.text().then(txt=>{
  const obj=JSON.parse(txt);
  let incoming;
  if(Array.isArray(obj.examples)) incoming=obj.examples;
  else if(Array.isArray(obj.coefficients)) incoming=[obj];
  else throw Error('Import expects {examples:[{coefficients:...}]} or a single residual object');
  if(incoming.length>250)throw Error('Import exceeds 250 examples');
  const clean=incoming.map((e,i)=>{
   const b=P.parseCoeffs(e.coefficients);
   return {id:'import-'+Date.now()+'-'+i,title:String(e.title||'Imported '+(i+1)).slice(0,120),
    coefficients:b.map(P.repr),strict:!!e.strict,description:String(e.description||'Imported polynomial model only.').slice(0,1500),
    family:String(e.family||'generic').slice(0,40)};
  });
  library=defaults.concat(clean);
  loadSelect();loadExample(clean[0]);
 }).catch(e=>{$('validation').textContent='Import rejected: '+e.message;$('validation').className='alert';});
}
$('preset').addEventListener('change',()=>loadExample(selectedExample()));
$('run').addEventListener('click',run);
$('depth').addEventListener('input',()=>{$('depthValue').textContent=$('depth').value;});
$('nodes').addEventListener('input',()=>{$('nodesValue').textContent=$('nodes').value;});
$('prev').addEventListener('click',()=>{stop();cursor=Math.max(0,cursor-1);focus=cursor-1;render();});
$('next').addEventListener('click',()=>{stop();cursor=Math.min(proof.nodes.length,cursor+1);focus=cursor-1;render();});
$('reset').addEventListener('click',()=>{stop();cursor=0;focus=-1;render();});
$('play').addEventListener('click',()=>{
 if(playback){stop();return;}
 if(!proof)return;
 if(cursor>=proof.nodes.length){cursor=0;focus=-1;}
 $('play').textContent='Ⅱ Pause';
 playback=setInterval(()=>{
  if(cursor>=proof.nodes.length){stop();return;}
  cursor++;focus=cursor-1;render();
 },430);
});
$('polygon').addEventListener('change',plot);
$('probe').addEventListener('input',plot);
$('export').addEventListener('click',()=>{
 if(!proof)return;
 const json={schema_version:1,scope:'SINGLE_EXACT_ENTERED_RESIDUAL_NOT_NATIVE_CERTIFICATE',
  title:current?current.title:'Custom',family:current?current.family:'generic',
  strict:proof.strict,coefficients:proof.original,proof:P.serializeProof(proof)};
 download('bernstein-proof-example.json',JSON.stringify(json,null,2));
});
$('importFile').addEventListener('change',e=>{fileImport(e.target.files[0]);e.target.value='';});
$('coeffs').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')run();});
loadSelect();loadExample(defaults[0]);
})();