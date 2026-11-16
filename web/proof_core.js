/* Exact-rational browser playground for a SINGLE Bernstein residual r(u).
 * Certificates here refer ONLY to the user-entered mathematical polynomial.
 * They are NOT native cuRobo/FP32 certificates or production proof replay.
 * No external libraries and no floating point in proof decisions.
 */
(function (global, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.BernsteinProof = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const ZERO = {n:0n,d:1n}, ONE = {n:1n,d:1n};
  function abs(a){ return a < 0n ? -a : a; }
  function gcd(a,b){ a=abs(a); b=abs(b); while(b){ const t=a%b; a=b; b=t; } return a || 1n; }
  function R(n,d=1n) {
    n=BigInt(n); d=BigInt(d);
    if (d===0n) throw Error('Zero denominator');
    if (d<0n) { n=-n; d=-d; }
    if (n===0n) return ZERO;
    const g=gcd(n,d); return {n:n/g,d:d/g};
  }
  function parse(v) {
    if (v && typeof v==='object' && typeof v.n==='bigint' && typeof v.d==='bigint') return R(v.n,v.d);
    let s=String(v).trim();
    if (s.length>128) throw Error('Rational input is too long');
    const slash=s.indexOf('/');
    if(slash>=0) {
      if(s.indexOf('/',slash+1)>=0) throw Error('Invalid fraction: '+s);
      return div(parse(s.slice(0,slash)),parse(s.slice(slash+1)));
    }
    const m=/^([+-]?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/.exec(s);
    if(!m) throw Error('Expected integer, decimal or fraction: '+s);
    const decimals=m[3]||'';
    const power=Number(m[4]||0)-decimals.length;
    if(Math.abs(power)>100) throw Error('Exponent is too large');
    let n=BigInt(m[2]+decimals);
    if(m[1]==='-') n=-n;
    return power>=0 ? R(n*10n**BigInt(power)) : R(n,10n**BigInt(-power));
  }
  function add(a,b){return R(a.n*b.d+b.n*a.d,a.d*b.d);}
  function neg(a){return R(-a.n,a.d);}
  function sub(a,b){return add(a,neg(b));}
  function mul(a,b){return R(a.n*b.n,a.d*b.d);}
  function div(a,b){if(b.n===0n)throw Error('Division by zero');return R(a.n*b.d,a.d*b.n);}
  function cmp(a,b){const v=a.n*b.d-b.n*a.d;return v>0n?1:v<0n?-1:0;}
  function min(v){return v.reduce((a,b)=>cmp(a,b)<0?a:b);}
  function max(v){return v.reduce((a,b)=>cmp(a,b)>0?a:b);}
  function eq(a,b){return cmp(a,b)===0;}
  function repr(a){return a.d===1n?String(a.n):String(a.n)+'/'+String(a.d);}
  function bitlen(x){x=abs(x);return x===0n?0:x.toString(2).length;}
  function num(a){ // VISUALIZATION ONLY: scale huge numerators/denominators together.
    let n=a.n,d=a.d;
    const excess=Math.max(bitlen(n),bitlen(d))-1000;
    if(excess>0){n>>=BigInt(excess);d>>=BigInt(excess);if(d===0n)return n<0n?-Infinity:Infinity;}
    return Number(n)/Number(d);
  }
  function split(b) {
    const rows=[b.slice()];
    while(rows[rows.length-1].length>1) {
      const prev=rows[rows.length-1], next=[];
      for(let i=0;i<prev.length-1;i++) next.push(div(add(prev[i],prev[i+1]),R(2)));
      rows.push(next);
    }
    const n=b.length-1;
    return [rows.map(row=>row[0]),Array.from({length:n+1},(_,i)=>rows[n-i][i])];
  }
  function at(b,t) {
    let a=b.slice();
    for(let k=a.length-1;k>0;k--) {
      const next=[];
      for(let i=0;i<k;i++) next.push(add(mul(sub(ONE,t),a[i]),mul(t,a[i+1])));
      a=next;
    }
    return a[0];
  }
  function sample(b,t) { // VISUALIZATION ONLY; never used in certification decisions.
    let a=b.map(num);
    for(let k=a.length-1;k>0;k--) a=Array.from({length:k},(_,i)=>a[i]*(1-t)+a[i+1]*t);
    return a[0];
  }

  function peakBracket(coeffs,depth=5) {
    // Exact mathematical global max bracket. Each leaf's Bernstein hull
    // upper-bounds the continuous peak, and evaluated dyadic points lower-bound it.
    // NOT a native CUDA error bound and NOT a complete K3 proof.
    const b=coeffs.map(parse);
    if(!Number.isInteger(depth)||depth<0||depth>10)throw Error('Invalid peak bracket depth');
    let leaves=[b], lower=cmp(b[0],b[b.length-1])>=0?b[0]:b[b.length-1];
    const initialUpper=max(b);
    for(let k=0;k<depth;k++) {
      const next=[];
      for(const leaf of leaves) next.push(...split(leaf));
      leaves=next;
    }
    let upper=null;
    for(const leaf of leaves){
      const hi=max(leaf);
      upper=upper===null||cmp(hi,upper)>0?hi:upper;
      const center=split(leaf)[0].slice(-1)[0];
      for(const value of [leaf[0],leaf[leaf.length-1],center])
        if(cmp(value,lower)>0)lower=value;
    }
    const gapLower=cmp(initialUpper,upper)>0?sub(initialUpper,upper):ZERO;
    const gapUpper=cmp(initialUpper,lower)>0?sub(initialUpper,lower):ZERO;
    return {scope:'EXACT_POLYNOMIAL_ONLY',depth,initialUpper,lower,upper,
      gapLower,gapUpper,leafCount:leaves.length};
  }

  function parseCoeffs(values) {
    if(!Array.isArray(values)||values.length<2||values.length>17) throw Error('Require 2–17 coefficients (degree 1–16)');
    return values.map(parse);
  }
  function strictWitness(node,strictLeft=false,strictRight=false) {
    const b=node.b;
    if((cmp(node.l,ZERO)>0||strictLeft) && cmp(b[0],ZERO)===0) return {at:node.l,value:b[0],kind:'ZERO'};
    if((cmp(node.r,ONE)<0||strictRight) && cmp(b[b.length-1],ZERO)===0) return {at:node.r,value:b[b.length-1],kind:'ZERO'};
    const m=div(add(node.l,node.r),R(2));
    const v=split(b)[0].slice(-1)[0];
    return cmp(v,ZERO)===0 ? {at:m,value:v,kind:'ZERO'} : null;
  }
  function build(coeffs, options={}) {
    const b=coeffs.map(parse);
    if(b.length<2||b.length>17)throw Error('Require 2–17 Bernstein coefficients');
    const depth=options.depth===undefined?8:Number(options.depth), nodesBudget=options.nodes===undefined?2048:Number(options.nodes);
    const strict=!!options.strict;
    // strictLeft/strictRight: also require r<0 at s=0 / s=1 (e.g. a span end that is an
    // interior knot of the trajectory). Only meaningful together with strict.
    const strictLeft=strict&&!!options.strictLeft, strictRight=strict&&!!options.strictRight;
    const witnesses=options.witnesses===undefined?true:!!options.witnesses;
    if(!Number.isInteger(depth)||depth<0||depth>24||!Number.isInteger(nodesBudget)||nodesBudget<1||nodesBudget>32768)throw Error('Invalid proof budget');
    const queue=[{path:'',l:ZERO,r:ONE,b,level:0}], nodes=[], leaves=[];
    let witness=null, status='CERTIFIED', processed=0, depthLimited=0;
    while(queue.length) {
      if(processed>=nodesBudget) {
        status='UNKNOWN_NODES';
        for(const item of queue) nodes.push({...item,lower:min(item.b),upper:max(item.b),state:'PENDING'});
        break;
      }
      const node=queue.shift();
      processed++;
      node.lower=min(node.b); node.upper=max(node.b);
      const midpoint=div(add(node.l,node.r),R(2));
      const midpointValue=split(node.b)[0].slice(-1)[0];
      let found=null;
      if(!witnesses){}
      else if(cmp(node.b[0],ZERO)>0) found={at:node.l,value:node.b[0],kind:'POSITIVE'};
      else if(cmp(node.b[node.b.length-1],ZERO)>0) found={at:node.r,value:node.b[node.b.length-1],kind:'POSITIVE'};
      else if(cmp(midpointValue,ZERO)>0) found={at:midpoint,value:midpointValue,kind:'POSITIVE'};
      if(!found && strict && witnesses) found=strictWitness(node,strictLeft,strictRight);
      if(found) {
        node.state='VIOLATED'; witness={...found,path:node.path};
        nodes.push(node); status=found.kind==='ZERO'?'VIOLATED_STRICT':'VIOLATED';
        break;
      }
      const hullPass=cmp(node.upper,ZERO)<=0;
      // A nonzero nonpositive Bernstein polynomial is strictly negative in the OPEN
      // interval: all Bernstein basis weights are positive in the open interval.
      // Endpoints that must be strict are additionally required to be nonzero.
      const strictlyInside=!eq(node.lower,ZERO)
        &&!((cmp(node.l,ZERO)>0||strictLeft)&&eq(node.b[0],ZERO))
        &&!((cmp(node.r,ONE)<0||strictRight)&&eq(node.b[node.b.length-1],ZERO));
      if(hullPass && (!strict || strictlyInside)) {
        node.state='PROVEN';nodes.push(node);
        leaves.push({path:node.path,coefficients:node.b.map(repr),lower:repr(node.lower),upper:repr(node.upper)});
        continue;
      }
      // Depth exhaustion is UNKNOWN for this leaf, but remaining queued nodes are still
      // examined so that an exact witness elsewhere is not hidden by an early stop.
      if(node.level>=depth) { node.state='UNKNOWN_DEPTH';nodes.push(node);depthLimited++;continue; }
      node.state='SPLIT';nodes.push(node);
      const [left,right]=split(node.b);
      queue.push({path:node.path+'L',l:node.l,r:midpoint,b:left,level:node.level+1},
                 {path:node.path+'R',l:midpoint,r:node.r,b:right,level:node.level+1});
    }
    if(status==='CERTIFIED'&&depthLimited)status='UNKNOWN_DEPTH';
    const opts={strict,strictLeft,strictRight};
    return {scope:'EXACT_RATIONAL_ENTERED_POLYNOMIAL_ONLY',status,strict,strictLeft,strictRight,witnesses,
      degree:b.length-1,maxDepth:depth,maxNodes:nodesBudget,processed,depthLimited,leaves,nodes,
      witness,original:b.map(repr),replay:status==='CERTIFIED'?replay(b,leaves,opts):{pass:false,reason:'No complete certificate'}};
  }
  const MAX_PATH=24, MAX_LEAVES=32768;
  function replay(root,leaves,options=false) {
    // Structural replay with SAME rational arithmetic; not an independent checker.
    const o=typeof options==='object'&&options?options:{strict:!!options};
    const strict=!!o.strict, strictLeft=strict&&!!o.strictLeft, strictRight=strict&&!!o.strictRight;
    try {
      root=root.map(parse);
      if(root.length<2||root.length>17) return {pass:false,reason:'Require 2-17 root coefficients'};
      if(!Array.isArray(leaves)||!leaves.length) return {pass:false,reason:'Empty leaf set'};
      if(leaves.length>MAX_LEAVES) return {pass:false,reason:'Too many leaves'};
      const paths=new Map();
      for(const leaf of leaves) {
        if(!leaf||typeof leaf.path!=='string'||!/^[LR]*$/.test(leaf.path)||leaf.path.length>MAX_PATH||paths.has(leaf.path))return {pass:false,reason:'Malformed, too deep or duplicate path'};
        if(!Array.isArray(leaf.coefficients))return {pass:false,reason:'Missing leaf coefficients'};
        let b=root.slice(),l=ZERO,r=ONE;
        for(const dir of leaf.path){
          const mid=div(add(l,r),R(2));
          const pair=split(b);
          if(dir==='L'){b=pair[0];r=mid;}else{b=pair[1];l=mid;}
        }
        const expect=b.map(repr);
        if(JSON.stringify(expect)!==JSON.stringify(leaf.coefficients))return {pass:false,reason:'Leaf coefficient mismatch'};
        if(cmp(max(b),ZERO)>0)return {pass:false,reason:'Leaf hull exceeds zero'};
        if(strict){
          if(eq(min(b),ZERO))return {pass:false,reason:'Identically zero leaf'};
          if((cmp(l,ZERO)>0||strictLeft) && eq(b[0],ZERO))return {pass:false,reason:'Zero at a strict left endpoint'};
          if((cmp(r,ONE)<0||strictRight) && eq(b[b.length-1],ZERO))return {pass:false,reason:'Zero at a strict right endpoint'};
        }
        paths.set(leaf.path,true);
      }
      for(const p of paths.keys())
        for(let k=0;k<p.length;k++) if(paths.has(p.slice(0,k)))return {pass:false,reason:'Overlapping leaves (a leaf is an ancestor of another)'};
      function full(p,depth=0){
        if(paths.has(p))return true;
        if(depth>MAX_PATH)return false;
        return full(p+'L',depth+1)&&full(p+'R',depth+1);
      }
      if(!full(''))return {pass:false,reason:'Leaves do not cover [0,1]'};
      return {pass:true,reason:'Exact leaf coefficients, hulls and dyadic coverage replayed (shared arithmetic)'};
    }catch(e){return {pass:false,reason:String(e.message||e)};}
  }
  function serializeProof(proof) {
    return JSON.parse(JSON.stringify(proof,(_,v)=>{
      if(v&&typeof v==='object'&&typeof v.n==='bigint'&&typeof v.d==='bigint'&&Object.keys(v).length===2)return repr(v);
      return typeof v==='bigint'?v.toString():v;
    }));
  }
  function uniform(coeffs,level) {
    // Uniform subdivision comparator: all 2^level dyadic cells, no early exit, no witnesses.
    const b=coeffs.map(parse);
    if(!Number.isInteger(level)||level<0||level>12)throw Error('Invalid uniform level');
    let cells=[b];
    for(let k=0;k<level;k++){const next=[];for(const c of cells)next.push(...split(c));cells=next;}
    const failing=cells.filter(c=>cmp(max(c),ZERO)>0).length;
    return {level,cells:cells.length,failing,status:failing?'UNKNOWN_UNIFORM':'CERTIFIED',
      worstUpper:max(cells.map(max))};
  }
  return {R,parse,add,sub,mul,div,cmp,eq,min,max,repr,num,at,sample,split,parseCoeffs,peakBracket,build,replay,serializeProof,uniform,MAX_PATH};
});