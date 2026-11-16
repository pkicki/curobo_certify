'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const P = require('../web/proof_core.js');
const B = values => values.map(P.parse);

test('exact decimal and rational input, normalized',()=>{
 assert.equal(P.repr(P.parse('0.1')), '1/10');
 assert.equal(P.repr(P.parse('-2/6')), '-1/3');
 assert.equal(P.repr(P.parse('1.25e-2')), '1/80');
 assert.equal(P.repr(P.add(P.parse('1/3'),P.parse('1/6'))),'1/2');
});
test('bad input rejected, no infinity/NaN or zero divisor',()=>{
 for(const bad of ['NaN','Infinity','1/0','foo','1/2/3','1e1000']){
  assert.throws(()=>P.parse(bad),bad);
 }
 assert.throws(()=>P.parseCoeffs(['1']));
 assert.throws(()=>P.parseCoeffs(Array(18).fill('1')));
});
test('exact midpoint subdivision preserves source polynomial',()=>{
 const b=B(['-1','2/5','-1']);
 const [l,r]=P.split(b);
 assert.equal(P.repr(P.at(b,P.parse('1/2'))),'-3/10');
 for(const u of ['0','1/4','1/3','1/2','3/4','1']){
  const t=P.parse(u);
  const v=P.at(b,t);
  const child=P.cmp(t,P.parse('1/2'))<=0?P.at(l,P.mul(t,P.R(2))):P.at(r,P.sub(P.mul(t,P.R(2)),P.R(1)));
  assert.equal(P.repr(child),P.repr(v));
 }
});
test('positive control coefficient can coexist with a fully feasible curve',()=>{
 const p=P.build(['-1','2/5','-1']);
 assert.equal(p.status,'CERTIFIED');
 assert.equal(p.leaves.length,2);
 assert.equal(p.nodes[0].state,'SPLIT');
 assert.equal(p.replay.pass,true);
});
test('depth zero is UNKNOWN, never incorrectly VIOLATED or CERTIFIED',()=>{
 const p=P.build(['-1','2/5','-1'],{depth:0});
 assert.equal(p.status,'UNKNOWN_DEPTH');
 assert.equal(p.leaves.length,0);
});
test('insufficient node budget is UNKNOWN',()=>{
 const p=P.build(['-1','2/5','-1'],{nodes:1});
 assert.equal(p.status,'UNKNOWN_NODES');
 assert.equal(p.replay.pass,false);
});
test('positive midpoint produces rational violation witness',()=>{
 const p=P.build(['-1','2','-1']);
 assert.equal(p.status,'VIOLATED');
 assert.equal(P.repr(p.witness.at),'1/2');
 assert.equal(P.repr(p.witness.value),'1/2');
});
test('strict interior tangency is a proven strict violation',()=>{
 const b=['-1/4','1/4','-1/4'];
 const nonstrict=P.build(b),strict=P.build(b,{strict:true});
 assert.equal(nonstrict.status,'CERTIFIED');
 assert.equal(strict.status,'VIOLATED_STRICT');
 assert.equal(P.repr(strict.witness.value),'0');
 assert.equal(P.repr(strict.witness.at),'1/2');
});
test('zero polynomial is okay nonstrict but fails strict',()=>{
 assert.equal(P.build(['0','0']).status,'CERTIFIED');
 assert.equal(P.build(['0','0'],{strict:true}).status,'VIOLATED_STRICT');
});
test('strict boundary zero permitted when interior is negative',()=>{
 const p=P.build(['0','-1','0'],{strict:true});
 assert.equal(p.status,'CERTIFIED');
 assert.equal(p.replay.pass,true);
});
test('high-degree synthetic jerk-style polynomial may be proven',()=>{
 const b=['-1/5','-1/5','-1/5','-1/5','1/10','1/10','-1/5','-1/5','-1/5','-1/5'];
 const p=P.build(b,{depth:12,nodes:2048});
 assert.equal(p.status,'CERTIFIED');
 assert.equal(p.replay.pass,true);
});
test('replay detects edited leaf coefficients',()=>{
 const b=B(['-1','2/5','-1']),p=P.build(b);
 p.leaves[0].coefficients[0]='-2';
 assert.equal(P.replay(b,p.leaves).pass,false);
});
test('replay detects incomplete dyadic coverage',()=>{
 const b=B(['-1','2/5','-1']),p=P.build(b);
 assert.equal(P.replay(b,p.leaves.slice(0,1)).pass,false);
});
test('replay rejects duplicates and malformed paths',()=>{
 const b=B(['-1','2/5','-1']),p=P.build(b);
 assert.equal(P.replay(b,[p.leaves[0],p.leaves[0]]).pass,false);
 assert.equal(P.replay(b,[{...p.leaves[0],path:'X'}]).pass,false);
});
test('exact proof outcome is independent of visualization sampling',()=>{
 const p=P.build(['-1','2/5','-1']);
 assert.equal(p.replay.pass,true);
 for(let i=0;i<=100;i++) assert.ok(P.sample(B(['-1','2/5','-1']),i/100)<=0);
});
test('serialized proof explicitly marks mathematical-only scope',()=>{
 const p=P.build(['-1','2/5','-1']);
 const j=P.serializeProof(p);
 assert.equal(j.scope,'EXACT_RATIONAL_ENTERED_POLYNOMIAL_ONLY');
 assert.equal(j.status,'CERTIFIED');
});

test('global maximum bracket and hull gap are rigorous for exact polynomial',()=>{
 const b=B(['-1','2/5','-1']);
 const t=P.peakBracket(b,5);
 assert.equal(P.repr(t.initialUpper),'2/5');
 assert.equal(P.repr(t.lower),'-3/10');
 assert.ok(P.cmp(t.upper,t.lower)>=0);
 assert.ok(P.cmp(t.upper,P.R(0))<0);
 assert.ok(P.cmp(t.gapLower,P.R(0))>0);
 for(let i=0;i<=100;i++){
   const y=P.at(b,P.R(BigInt(i),100n));
   assert.ok(P.cmp(y,t.lower)<=0);
   assert.ok(P.cmp(y,t.upper)<=0);
 }
});

// ---- Regression tests added 2026-11-16 (defects found during Proof Lab audit) ----
test('strict endpoint flags: zero at an interior trajectory knot is a strict violation',()=>{
 // -x(s) for x(s)=s on a middle span: x vanishes at the span's left end.
 const b=['0','-1/2','-1'];
 assert.equal(P.build(b,{strict:true}).status,'CERTIFIED');
 const p=P.build(b,{strict:true,strictLeft:true});
 assert.equal(p.status,'VIOLATED_STRICT');
 assert.equal(P.repr(p.witness.at),'0');
 assert.equal(P.build(b,{strict:true,strictRight:true}).status,'CERTIFIED');
});
test('strict endpoint flags are enforced by replay and without witness search',()=>{
 const b=B(['0','-1/2','-1']);
 const ok=P.build(b,{strict:true});
 assert.equal(P.replay(b,ok.leaves,{strict:true}).pass,true);
 assert.equal(P.replay(b,ok.leaves,{strict:true,strictLeft:true}).pass,false);
 const nw=P.build(b,{strict:true,strictLeft:true,witnesses:false,depth:4});
 assert.equal(nw.status,'UNKNOWN_DEPTH');
});
test('depth exhaustion does not hide a witness in a later queued node',()=>{
 // Found by exhaustive search over small integer quintics: at depth limit 1 the left child
 // is inconclusive (UNKNOWN_DEPTH) and the right child has an exact positive witness.
 const c=['-1','-1','3','-4','3','-1'];
 const p=P.build(c,{depth:1});
 assert.equal(p.nodes.find(n=>n.path==='L').state,'UNKNOWN_DEPTH');
 assert.equal(p.status,'VIOLATED');
 assert.equal(p.witness.path,'R');
 assert.ok(P.cmp(P.at(B(c),p.witness.at),P.R(0))>0);
 // Without witness handling the same budget can only say UNKNOWN.
 assert.equal(P.build(c,{depth:1,witnesses:false}).status,'UNKNOWN_DEPTH');
});
test('replay rejects overlapping leaves and over-deep paths',()=>{
 const b=B(['-1','2/5','-1']),p=P.build(b);
 const root={path:'',coefficients:b.map(P.repr)};
 // root alone is not provable, so overlap must be detected structurally with a provable root
 const s=B(['-1','-1']), ps=P.build(s);
 assert.equal(P.replay(s,ps.leaves).pass,true);
 const [lc,rc]=P.split(s);
 const extra={path:'L',coefficients:lc.map(P.repr)};
 assert.equal(P.replay(s,ps.leaves.concat([extra])).pass,false);
 assert.equal(P.replay(b,[{path:'L'.repeat(25),coefficients:['0','0','0']}]).pass,false);
 assert.equal(P.replay(b,[{path:'L'}]).pass,false); // missing coefficients
 assert.ok(root);
});
test('without early witness handling a genuine violation is only UNKNOWN',()=>{
 const p=P.build(['-1','2','-1'],{witnesses:false,depth:6});
 assert.equal(p.status,'UNKNOWN_DEPTH');
 assert.equal(p.witness,null);
});
test('uniform comparator matches adaptive outcome on false-alarm example',()=>{
 assert.equal(P.uniform(['-1','2/5','-1'],0).status,'UNKNOWN_UNIFORM');
 const u1=P.uniform(['-1','2/5','-1'],1);
 assert.equal(u1.status,'CERTIFIED');assert.equal(u1.cells,2);
});
test('plot conversion handles huge numerators and denominators',()=>{
 const big=P.R(10n**400n+1n,3n*10n**400n);
 assert.ok(Math.abs(P.num(big)-1/3)<1e-12);
 assert.equal(P.num(P.R(10n**400n,1n)),Infinity);
});
test('serialized proof uses exact rational strings',()=>{
 const j=P.serializeProof(P.build(['-1','2/5','-1']));
 assert.equal(j.nodes[0].l,'0');assert.equal(j.nodes[0].upper,'2/5');
 assert.equal(j.nodes[1].r,'1/2');
});
test('seeded exhaustive soundness: CERTIFIED never contradicted by exact dyadic evaluation',()=>{
 let seed=20261116;
 const rnd=()=>{seed=(seed*1103515245+12345)%2147483648;return seed;};
 let certified=0,violated=0;
 for(let trial=0;trial<300;trial++){
  const deg=1+rnd()%6;
  const c=Array.from({length:deg+1},()=>P.R(BigInt((rnd()%41)-30),BigInt(1+rnd()%7)));
  const strict=rnd()%2===0;
  const p=P.build(c,{depth:6,nodes:512,strict});
  if(p.status==='CERTIFIED'){
   certified++;
   assert.equal(p.replay.pass,true);
   for(let i=0;i<=64;i++){
    const y=P.at(c,P.R(BigInt(i),64n));
    assert.ok(P.cmp(y,P.R(0))<=0);
    if(strict&&i>0&&i<64)assert.ok(P.cmp(y,P.R(0))<0);
   }
  }
  if(p.status.startsWith('VIOLATED')){
   violated++;
   const y=P.at(c,p.witness.at);
   assert.equal(P.repr(y),P.repr(p.witness.value));
   assert.ok(p.status==='VIOLATED'?P.cmp(y,P.R(0))>0:P.cmp(y,P.R(0))===0);
  }
 }
 assert.ok(certified>20&&violated>20,`certified=${certified} violated=${violated}`);
});
