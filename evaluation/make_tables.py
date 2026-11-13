"""Prints the six per-domain accuracy tables and the pass@k summary in LaTeX.

Each cell is "c/5" (correct samples under the Correctness criterion, with a
dagger when the greedy pass is also correct) followed by the Domain
Coverage, taken as the best of the five samples of the cell.

Usage: python3 evaluation/make_tables.py   (after evaluation/grade_all.sh)
"""
import json, glob, os, sys
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, "graded")
rep = {json.load(open(p))['file']: json.load(open(p)) for p in glob.glob(G + '/repaired/*.json')}
anyd = {json.load(open(p))['file']: json.load(open(p)) for p in glob.glob(G + '/coverage/*.json')}
if len(rep) < 1100:
    raise SystemExit("run evaluation/grade_all.sh first (found %d graded responses)" % len(rep))
# a response whose grader was killed leaves no file and counts as a failure

MODELS = [("deepseek-r1:70b","R1:70b"),("gemma3:12b","Gem3:12b"),("gemma3:27b","Gem3:27b"),
          ("gpt-oss:120b","OSS:120b"),("gpt-oss:20b","OSS:20b"),("llama3.3","Lla3.3:70b"),
          ("llama4","Lla4:16x17b"),("mistral-nemo","Mist-N:12b"),("nemotron:70b","Nemo:70b"),
          ("qwen3:235b","Qw3:235b"),("qwen3:32b","Qw3:32b")]
DOMS = [("2D_Triangular","tab:2d_triangular","2D Triangular mapping","validation"),
        ("2D_Triangulo_Sierpinski","tab:2d_gasket","2D Sierpinski gasket","validation"),
        ("2D_Alfombra_Sierpinski","tab:2d_carpet","2D Sierpinski carpet","validation"),
        ("3D_Triangular","tab:3d_triangular","3D triangular (tetrahedral) mapping","validation"),
        ("3D_Piramide_Sierpinski","tab:3d_pyramid","3D Sierpinski pyramid","discovery"),
        ("3D_Menger_Sponge","tab:3d_menger","3D Menger sponge","discovery")]
STAGES = (20, 50, 100)

def key(d, m, s, r): return "%s__%s__st%d__%s" % (d, m.replace(':','_'), s, r)
def cval(d, m, s): return sum(1 for r in "12345" if rep.get(key(d,m,s,r), {}).get('passed'))
def greedy(d, m, s): return bool(rep.get(key(d,m,s,'greedy'), {}).get('passed'))
def anyo(d, m, s):
    vs = [anyd[key(d,m,s,r)]['any_order'] for r in "12345" if key(d,m,s,r) in anyd]
    vs = [v for v in vs if v is not None]
    return max(vs) if vs else None
def pk(c, n, k): return 1.0 if n-c < k else 1.0 - comb(n-c,k)/comb(n,k)

def cell(d, m, s):
    c = cval(d,m,s); g = greedy(d,m,s); a = anyo(d,m,s)
    body = r"\textbf{%d/5}" % c if c else "0/5"
    if g: body += r"$^{\dagger}$"
    aa = "---" if a is None else (r"%.1f\%%" % (100*a))
    return "%s & %s" % (body, aa)

out = []
for dom, lab, name, kind in DOMS:
    L = []
    L.append(r"\begin{table}[pos=htbp]")
    L.append(r"\centering")
    L.append(r"\caption{%s (%s set). The \textit{Correctness} column gives the number" % (name[0].upper()+name[1:], kind))
    L.append(r"of the five sampled responses that reproduce the mapping exactly over $10^{6}$")
    L.append(r"indices, which is pass@1 of Section~\ref{subsec:passk}; $^{\dagger}$ marks a cell")
    L.append(r"whose greedy ($T=0$) pass also succeeds. \textit{Domain Coverage} is the percentage")
    L.append(r"of the domain's distinct coordinates covered by the map regardless of order, taking")
    L.append(r"the best of the five samples and not their average.}")
    L.append(r"\label{%s}" % lab)
    L.append(r"\small")
    L.append(r"\begin{tabular}{lcccccc}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{20 pts}} & \multicolumn{2}{c}{\textbf{50 pts}} & \multicolumn{2}{c}{\textbf{100 pts}} \\ \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    L.append(r" & \textbf{Correctness} & \textbf{Domain Coverage} & \textbf{Correctness} & \textbf{Domain Coverage} & \textbf{Correctness} & \textbf{Domain Coverage} \\ \midrule")
    for m, ab in MODELS:
        L.append("%s & %s & %s & %s \\\\" % (ab, cell(dom,m,20), cell(dom,m,50), cell(dom,m,100)))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    out.append((lab, "\n".join(L)))

# summary table
S = []
S.append(r"\begin{table}[pos=htbp]")
S.append(r"\centering")
S.append(r"\caption{pass@$k$ by model, averaged over the 18 (domain, stage) cells.")
S.append(r"$n=5$ samples per cell; greedy is a separate $T=0$ pass. Correctness criterion.}")
S.append(r"\label{tab:passk_summary}")
S.append(r"\small")
S.append(r"\begin{tabular}{lrrrr}")
S.append(r"\toprule")
S.append(r"\textbf{Model} & \textbf{pass@1} & \textbf{pass@3} & \textbf{pass@5} & \textbf{Greedy} \\")
S.append(r"\midrule")
rows = []
for m, ab in MODELS:
    p1 = p3 = p5 = 0.0; g = 0
    for dom,_,_,_ in DOMS:
        for s in STAGES:
            c = cval(dom,m,s); g += greedy(dom,m,s)
            p1 += pk(c,5,1); p3 += pk(c,5,3); p5 += pk(c,5,5)
    rows.append((ab, 100*p1/18, 100*p3/18, 100*p5/18, 100*g/18))
for ab,a,b,c,d in sorted(rows, key=lambda x: -x[3]):
    bold = (lambda v: r"\textbf{%.1f\%%}" % v) if ab == "OSS:120b" else (lambda v: r"%.1f\%%" % v)
    S.append("%s & %s & %s & %s & %s \\\\" % (
        (r"\textbf{%s}" % ab) if ab=="OSS:120b" else ab, bold(a), bold(b), bold(c), bold(d)))
S.append(r"\bottomrule")
S.append(r"\end{tabular}")
S.append(r"\end{table}")
out.append(("tab:passk_summary", "\n".join(S)))

for k, v in out: print("%%%%%%%% " + k + "\n" + v + "\n")
