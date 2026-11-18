import sys, tempfile, random, json, time, hashlib, numpy as np
sys.path[:0] = ["src", "."]
from pathlib import Path
from curobo_certify import k4_compiled as KC
from curobo_certify.k4_panda_model import PandaModel, load_contract
from independent_checks.k4p2 import campaign as C
from tests.test_independent_k4_p5b import CP
t0=time.time()
c=load_contract('docs/reports/k4/p0/panda_model_contract_v2.json'); M=PandaModel(c)
d=Path(tempfile.mkdtemp()); bp=KC.build(d/'k.so'); bv=KC.build_verifier(d/'v.so')
P=KC.CompiledPath(M,d/'k.so'); V=KC.CompiledVerifier(c,d/'v.so'); orc=C.oracle('iv',c)
lo=np.array([-87.]*4+[-12.]*3); hi=-lo
rng=random.Random(20261118); stat={}; ver={}; verd={}; bad=[]; det_mismatch=0; forged_accepted=0; forged_tried=0
for k in range(40):
    g,tc=CP.random_case(rng,neg_timing=(k%8==7))
    for f in (1.0,0.6,0.4):
        L,H=lo*f,hi*f
        r=P.certify(g,tc,L,H,max_depth=12,max_nodes=100000,threads=1)
        r8=P.certify(g,tc,L,H,max_depth=12,max_nodes=100000,threads=8)
        det_mismatch+= (r!=r8)
        stat[r['status']]=stat.get(r['status'],0)+1
        st,b,_=CP.check(orc,g,tc,r,L,H,random.Random(k),4)
        for kk,v in st.items(): verd[kk]=verd.get(kk,0)+v
        bad+=[dict(x,case=k,f=f) for x in b]
        if r['status']=='CERTIFIED':
            v=V.verify(g,tc,r,L,H,refine=8); ver[str(v['pass'])]=ver.get(str(v['pass']),0)+1
        if r['status']=='VIOLATED_TORQUE':
            ns=r['witness']['span']+1
            from curobo_certify.spline_polynomials import extract_timeopt_spans
            nsp=len(extract_timeopt_spans(g,tc))
            forged={'status':'CERTIFIED','leaves':[(s,'',list(L),list(H)) for s in range(nsp)],
                    'input_digest':KC.bytes_digest(g,tc,c['contract_sha256'],L,H)}
            forged_tried+=1; forged_accepted+=V.verify(g,tc,forged,L,H,refine=12)['pass']
out={'scope':'compiled pipeline vs independent Lagrangian oracle (falsification)','build_producer':bp,'build_verifier':bv,
     'cases':40,'scales':[1.0,0.6,0.4],'status':stat,'oracle_verdicts':verd,'counterexamples':bad,
     'verifier_on_certified':ver,'thread_1_vs_8_mismatches':det_mismatch,
     'forged_certificates_for_violated_paths':{'tried':forged_tried,'accepted':forged_accepted},'wall_s':round(time.time()-t0,1)}
json.dump(out,open('docs/reports/k4_p5b_independent_review/campaign.json','w'),indent=1,default=str)
print(json.dumps({k:v for k,v in out.items() if not k.startswith('build')}))
