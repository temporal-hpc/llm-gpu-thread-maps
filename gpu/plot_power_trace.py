"""Plots the power trace written by power_trace (ms, watts, phase).

The time axis is divided by the number of launches of each phase, so that
the width of each shaded span is the time of one kernel launch.

Usage: python3 gpu/plot_power_trace.py [data/power_trace.csv] [out.pdf]
"""
import csv, os, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "power_trace.csv")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "power_trace_2d_triangular.pdf")
# Each phase launches its kernel PHASE_LAUNCHES times (power_trace.cu). The x
# axis is divided by that number and reads in ms per launch.
LAUNCHES = 2000
ms, w, ph = [], [], []
with open(SRC) as f:
    for r in csv.DictReader(f):
        ms.append(float(r["ms"])/LAUNCHES); w.append(float(r["watts"])); ph.append(int(r["phase"]))

fig, ax = plt.subplots(figsize=(7.0, 2.9))
ax.plot(ms, w, lw=1.2, color="#1f3b73", zorder=3)

# shade the kernel phases
def spans(target):
    out, s = [], None
    for i, p in enumerate(ph):
        if p == target and s is None: s = ms[i]
        elif p != target and s is not None: out.append((s, ms[i])); s = None
    if s is not None: out.append((s, ms[-1]))
    return out
for a, b in spans(1):
    ax.axvspan(a, b, color="#d9534f", alpha=.13, zorder=1)
for a, b in spans(2):
    ax.axvspan(a, b, color="#4a8f4a", alpha=.13, zorder=1)

def mean(t): 
    v=[w[i] for i in range(len(w)) if ph[i]==t]; return sum(v)/len(v)
m0, m1, m2 = mean(0), mean(1), mean(2)
for m, c in ((m0,"#777"), (m1,"#d9534f"), (m2,"#4a8f4a")):
    ax.axhline(m, ls="--", lw=.8, color=c, zorder=2)

a,b = spans(1)[0]; t1=b-a; ax.text((a+b)/2, m1+22, f"bounding box\n{m1:.0f} W, {t1:.2f} ms, {m1*t1/1000:.2f} J", ha="center", fontsize=8.5, color="#a33")
a,b = spans(2)[0]; t2=b-a; ax.text((a+b)/2, m2+22, f"efficient GPU map\n{m2:.0f} W, {t2:.2f} ms, {m2*t2/1000:.2f} J", ha="center", fontsize=8.5, color="#2a662a")
ax.text(ms[0]+0.4/2, m0+16, f"idle {m0:.0f} W", fontsize=8, color="#555")

ax.set_xlabel("time per kernel launch (ms)"); ax.set_ylabel("GPU power (W)")
ax.set_xlim(0, ms[-1]); ax.set_ylim(0, max(w)*1.22)
ax.grid(alpha=.25, lw=.5); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(DST)
print(f"BB {t1:.3f} ms {m1*t1/1000:.4f} J | EFF {t2:.3f} ms {m2*t2/1000:.4f} J | t {t1/t2:.3f}x E {m1*t1/(m2*t2):.3f}x")
print(f"idle {m0:.1f} W | BB {m1:.1f} W | map {m2:.1f} W | power ratio {m2/m1:.2f}x | figure in {DST}")
