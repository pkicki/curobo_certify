// SPDX-License-Identifier: MIT
// Conservative binary64 interval hulls for local-power cubic q(u), x(u).
// Diagnostics/early screening ONLY: does not validate boundaries, time inversion,
// native FP32 evaluation, knot regularity, or rigorous independent proof replay.
// Compile with -fno-fast-math -ffp-contract=off, use IEEE-754 binary64 SSE.
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
struct I { double lo, hi; };
inline void check(const I &a) {
    if (!std::isfinite(a.lo) || !std::isfinite(a.hi) || a.lo > a.hi)
        throw std::range_error("nonfinite, reversed or overflowed interval");
}
inline double down(double x) { return std::nextafter(x, -std::numeric_limits<double>::infinity()); }
inline double up(double x) { return std::nextafter(x, std::numeric_limits<double>::infinity()); }
inline I exact(double x) { return {x,x}; }
inline bool is_zero(I a) {return a.lo == 0.0 && a.hi == 0.0;}
inline bool is_one(I a) {return a.lo == 1.0 && a.hi == 1.0;}
inline I add(I a, I b) {
    if(is_zero(a)) return b;
    if(is_zero(b)) return a;
    I c{down(a.lo+b.lo), up(a.hi+b.hi)};check(c);return c;
}
inline I neg(I a) { return {-a.hi,-a.lo}; }
inline I sub(I a,I b) { return add(a, neg(b)); }
inline I mul(I a, I b) {
    if(is_zero(a)||is_zero(b))return exact(0);
    if(is_one(a))return b;
    if(is_one(b))return a;
    const double v[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
    for(double x:v) if(!std::isfinite(x)) throw std::range_error("nonfinite product");
    const auto r = std::minmax_element(std::begin(v),std::end(v));
    I c{down(*r.first),up(*r.second)}; check(c); return c;
}
inline I div_positive(I a, I b) {
    if (b.lo <= 0 || !std::isfinite(b.lo) || !std::isfinite(b.hi))
        throw std::range_error("nonpositive denominator interval");
    if(is_zero(a))return exact(0);
    if(is_one(b))return a;
    const I inv{down(1.0/b.hi),up(1.0/b.lo)};check(inv);
    return mul(a,inv);
}
using P=std::vector<I>;
inline P padd(const P& a,const P& b) {
    P c(std::max(a.size(),b.size()),exact(0));
    for(std::size_t k=0;k<c.size();++k) c[k]=add(k<a.size()?a[k]:exact(0),k<b.size()?b[k]:exact(0));
    return c;
}
inline P pscale(const P& a,I s) {P c; c.reserve(a.size());for(auto v:a)c.push_back(mul(v,s));return c;}
inline P pmul(const P& a,const P& b) {
    if(a.empty()||b.empty())throw std::range_error("empty polynomial");
    P c(a.size()+b.size()-1,exact(0));
    for(std::size_t i=0;i<a.size();++i)
       for(std::size_t j=0;j<b.size();++j) c[i+j]=add(c[i+j],mul(a[i],b[j]));
    return c;
}
inline P pdiff(const P& a,I invh) {
    if (a.size()<2) return {exact(0)};
    P c; c.reserve(a.size()-1);
    for(std::size_t i=1;i<a.size();++i)c.push_back(mul(mul(a[i], exact(static_cast<double>(i))),invh));
    return c;
}
inline unsigned binom(unsigned n,unsigned k) {
    unsigned result=1;
    for(unsigned j=1;j<=k;++j)result=(result*(n-j+1))/j;
    return result;
}
inline I bernstein_hull(const P& power) {
    if(power.empty()||power.size()>10)throw std::range_error("degree not supported");
    const unsigned n=static_cast<unsigned>(power.size()-1);
    double lo=std::numeric_limits<double>::infinity(),hi=-lo;
    for(unsigned k=0;k<=n;++k) {
        I sum=exact(0);
        for(unsigned i=0;i<=k;++i) {
            I fraction=div_positive(exact(static_cast<double>(binom(k,i))),
                                    exact(static_cast<double>(binom(n,i))));
            sum=add(sum,mul(power[i],fraction));
        }
        lo=std::min(lo,sum.lo);hi=std::max(hi,sum.hi);
    }
    I result{lo,hi};check(result);return result;
}
inline I input(const double* lows,const double* highs,std::size_t n) {
    I v{lows[n],highs[n]};check(v);return v;
}
inline void compute(const double* qlo,const double* qhi,const double* xlo,const double* xhi,
                    const double* hlo,const double* hhi,
                    const double* vlo,const double* vhi,
                    const double* alo,const double* ahi,
                    const double* jlo,const double* jhi,
                    int ns,int nj,double* outputlo,double* outputhi) {
    if(ns<=0||nj<=0||ns>100000||nj>1000)
        throw std::range_error("invalid span or joint count");
    for(int s=0;s<ns;++s) {
        I h=input(hlo,hhi,s);
        if(h.lo<=0)throw std::range_error("nonpositive span width");
        I invh=div_positive(exact(1),h);
        P x;for(int k=0;k<4;++k)x.push_back(input(xlo,xhi,s*4+k));
        P xp=pdiff(x,invh),xpp=pdiff(xp,invh);
        const P xpos=pscale(x,exact(-1));
        I positivity=bernstein_hull(xpos);
        for(int j=0;j<nj;++j){
            P q;for(int k=0;k<4;++k)q.push_back(input(qlo,qhi,(s*nj+j)*4+k));
            P q1=pdiff(q,invh),q2=pdiff(q1,invh),q3=pdiff(q2,invh);
            I vl=input(vlo,vhi,j),al=input(alo,ahi,j),jl=input(jlo,jhi,j);
            if(vl.lo<=0||al.lo<=0||jl.lo<=0)
                throw std::range_error("nonpositive actuator limit");
            const I vinv=div_positive(exact(1),mul(vl,vl));
            const I ainv=div_positive(exact(1),al);
            const I jinv=div_positive(exact(1),mul(jl,jl));
            const P vres=padd(pscale(pmul(pmul(q1,q1),x),vinv),{exact(-1)});
            const P acc=padd(pmul(q2,x),pscale(pmul(q1,xp),exact(0.5)));
            const P aplus=padd(pscale(acc,ainv),{exact(-1)});
            const P aminus=padd(pscale(acc,neg(ainv)),{exact(-1)});
            const P ell=padd(padd(pmul(q3,x),pscale(pmul(q2,xp),exact(1.5))),
                             pscale(pmul(q1,xpp),exact(0.5)));
            const P jerk=padd(pscale(pmul(x,pmul(ell,ell)),jinv),{exact(-1)});
            const std::array<I,5> hulls={positivity,bernstein_hull(vres),
                bernstein_hull(aplus),bernstein_hull(aminus),bernstein_hull(jerk)};
            for(int f=0;f<5;++f) {
                std::size_t id=(static_cast<std::size_t>(s)*nj+j)*5+f;
                outputlo[id]=hulls[f].lo;outputhi[id]=hulls[f].hi;
            }
        }
    }
}
} // namespace

extern "C" {
// Return 0 on success; -1 invalid/nonfinite/unsupported input; -2 unknown FP
// environment. Never authorize a full certificate from this function alone.
int timeopt_k3_hull_v1(const double* qlo,const double* qhi,
    const double* xlo,const double* xhi,const double* hlo,const double* hhi,
    const double* vlo,const double* vhi,const double* alo,const double* ahi,
    const double* jlo,const double* jhi,int ns,int nj,
    double* outputlo,double* outputhi) noexcept {
    if(!std::numeric_limits<double>::is_iec559 || FLT_EVAL_METHOD!=0) return -2;
    if(!qlo||!qhi||!xlo||!xhi||!hlo||!hhi||!vlo||!vhi||!alo||!ahi||!jlo||!jhi||!outputlo||!outputhi)
        return -1;
    try {compute(qlo,qhi,xlo,xhi,hlo,hhi,vlo,vhi,alo,ahi,jlo,jhi,ns,nj,outputlo,outputhi);}
    catch(...) {return -1;}
    return 0;
}
}
