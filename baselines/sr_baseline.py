"""Symbolic-regression baseline (gplearn) on the 2D triangular domain.

Question: does a continuous symbolic regressor produce an exact integer
bijection for thread mapping, or only a numerical fit? Each coordinate is
fitted on the first 1,000 indices and evaluated over 10^6.

Requires: pip install gplearn numpy
Usage: python3 baselines/sr_baseline.py   (writes data/sr_baseline_results.json)
"""
import os, re, json, numpy as np
from gplearn.genetic import SymbolicRegressor

N_FIT, N_EVAL = 1000, 1_000_000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOM = os.path.join(ROOT, 'domains', '2D_Triangular.txt')

idx, xs, ys = [], [], []
with open(DOM) as f:
    for i, l in enumerate(f):
        if i >= N_EVAL: break
        m = re.findall(r'-?\d+', l)
        idx.append(int(m[0])); xs.append(int(m[1])); ys.append(int(m[2]))
lam = np.array(idx, float); X = np.array(xs, float); Y = np.array(ys, float)

FUNCS = ('add','sub','mul','div','sqrt','log','abs','neg','inv')
out, preds = {}, {}
for name, tgt in (('x', X), ('y', Y)):
    est = SymbolicRegressor(population_size=2000, generations=40,
                            function_set=FUNCS, parsimony_coefficient=0.001,
                            metric='mse', random_state=0, n_jobs=8, verbose=0)
    est.fit(lam[:N_FIT].reshape(-1,1), tgt[:N_FIT])
    p = est.predict(lam.reshape(-1,1)); preds[name] = p
    r2 = 1 - float(np.sum((tgt-p)**2))/float(np.sum((tgt-tgt.mean())**2))
    out[name] = dict(expr=str(est._program), r2=r2,
                     mse=float(np.mean((tgt-p)**2)),
                     exact_raw=float(np.mean(p == tgt)),
                     exact_rounded=float(np.mean(np.rint(p) == tgt)),
                     max_abs_err=float(np.max(np.abs(p-tgt))))
    o=out[name]
    print(f"[{name}] {o['expr']}")
    print(f"     R2={o['r2']:.6f}  MSE={o['mse']:.4g}  max_err={o['max_abs_err']:.4g}")
    print(f"     exact raw={o['exact_raw']*100:.4f}%   exact rounded={o['exact_rounded']*100:.4f}%")

# the rounded (x, y) pair: is it bijective over the domain?
px, py = np.rint(preds['x']).astype(np.int64), np.rint(preds['y']).astype(np.int64)
pairs = set(zip(px.tolist(), py.tolist()))
gt_pairs = set(zip(np.asarray(xs), np.asarray(ys)))
both = float(np.mean((px == X) & (py == Y)))
out['pair'] = dict(exact_both=both, unique_pairs=len(pairs), gt_pairs=len(gt_pairs),
                   covered=len(pairs & gt_pairs), negative=int(np.sum((px<0)|(py<0))))
print(f"\n[pair] both exact = {both*100:.4f}%")
print(f"      distinct coordinates emitted = {len(pairs):,} of {len(gt_pairs):,} in the domain")
print(f"      covered from the domain = {len(pairs & gt_pairs):,}")
print(f"      with a negative coordinate = {out['pair']['negative']:,}")
json.dump(out, open(os.path.join(ROOT, 'data', 'sr_baseline_results.json'), 'w'), indent=2)
print("\nresults in data/sr_baseline_results.json")
