"""Figures for the report, drawn only from results/*.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
R = json.loads((ROOT / "results" / "results.json").read_text())
K = json.loads((ROOT / "results" / "constraint.json").read_text())
FIG = ROOT / "results" / "figures"
FIG.mkdir(exist_ok=True)
INK, MUTED, ACCENT, GREY = "#1f2933", "#6b7280", "#b4472f", "#9aa5b1"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def fig_models():
    names = list(R["models"])
    order = sorted(names, key=lambda n: R["models"][n]["macro_f1"])
    vals = [R["models"][n]["macro_f1"] for n in order]
    cols = [ACCENT if n.startswith("TF-IDF narrative (with") else GREY for n in order]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.barh(order, vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, 0.47)
    ax.set_xlabel(f"Macro-F1 on {R['split']['test']} test year (3 classes)")
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIG / "fig_models.png"); plt.close(fig)


def fig_forest():
    rows = [("Crude (all crossing pedestrians)", R["or_crude"]),
            ("Adjusted (logistic regression)", R["or_adjusted"]["terms"]["masked"])]
    rows += [(f"Stratum: {k} mph limit", v) for k, v in R["or_by_speed_limit"].items() if k in ("20", "30", "40")]
    rows += [(f"Stratum: {k}", v) for k, v in R["or_by_age"].items()]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    for i, (lab, v) in enumerate(rows):
        main = lab.startswith(("Crude", "Adjusted"))
        c = ACCENT if main else INK
        ax.plot(v["ci95"], [i, i], color=c, lw=1.6)
        ax.plot(v["or"], i, "o", color=c, ms=5)
        ax.text(1.62, i, f"{v['or']:.2f} ({v['ci95'][0]:.2f}-{v['ci95'][1]:.2f})", va="center", fontsize=8.5)
    ax.axvline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xscale("log"); ax.set_xlim(0.5, 1.6)
    ax.set_xticks([0.5, 0.7, 1.0, 1.3, 1.6], ["0.5", "0.7", "1.0", "1.3", "1.6"]); ax.minorticks_off()
    ax.set_xlabel("Odds ratio of killed-or-seriously-injured, masked vs unmasked crossing (95% CI)")
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIG / "fig_forest.png"); plt.close(fig)


def fig_coverage():
    c = R["coverage"]
    other = 1 - c["on_facility_share"] - c["within_50m_share"]
    labs = ["On a crossing facility\n(sensors can see them)", "Within 50 m of a facility", "Elsewhere on the road"]
    vals = [c["on_facility_share"], c["within_50m_share"], other]
    fig, ax = plt.subplots(figsize=(6.4, 2.1))
    ax.barh(labs[::-1], [100 * v for v in vals[::-1]], color=[GREY, GREY, ACCENT], height=0.6)
    for i, v in enumerate(vals[::-1]):
        ax.text(100 * v + 1, i, f"{100 * v:.1f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, 90); ax.set_xlabel(f"Share of the {c['masked_total']:,} masked-crossing pedestrian casualties")
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIG / "fig_coverage.png"); plt.close(fig)


def fig_impact():
    a = K["assumptions"]
    import numpy as np
    from crossing_governor import impact_speed
    d = np.linspace(0, 45, 300)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for (lab, v), col, lw in zip([("15 km/h (cap)", 15.0), ("20 mph", 32.2), ("30 mph", 48.3), ("40 mph", 64.4)],
                                 [ACCENT, "#52606d", "#7b8794", "#9aa5b1"], [2.2, 1.4, 1.4, 1.4]):
        y = [impact_speed(v, s, a["reaction_s"], a["emergency_decel_ms2"]) for s in d]
        ax.plot(d, y, color=col, lw=lw)
        ax.text(0.4, v + 1.5, lab, color=col, fontsize=8.5)
    ax.set_xlabel("Distance at which the pedestrian first becomes visible (m)")
    ax.set_ylabel("Impact speed (km/h)"); ax.set_ylim(0, 72); ax.set_xlim(0, 45)
    fig.savefig(FIG / "fig_impact.png"); plt.close(fig)


def fig_control():
    fig, ax = plt.subplots(figsize=(6.4, 4.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(x, y, w, h, title, sub="", hot=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#fbeee9" if hot else "#f3f4f6", ec=ACCENT if hot else MUTED, lw=1))
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), title, ha="center", va="center", fontsize=8.5, weight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center", fontsize=7.2, color=MUTED)

    def arrow(x0, y0, x1, y1, txt, side=1, col=INK):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1))
        ax.text((x0 + x1) / 2 + 0.12 * side, (y0 + y1) / 2, txt, fontsize=7.2, color=col,
                ha="left" if side > 0 else "right", va="center")

    box(1.6, 8.7, 6.8, 1.0, "Highway authority / regulator", "posted limits, crossing standards, parking rules")
    box(1.6, 6.4, 8.2, 1.1, "Crossing safety controller (roadside unit)", "the constraint in crossing_governor.py", hot=True)
    box(0.2, 3.6, 4.5, 1.1, "Vehicle speed controller", "speed limiter / automated driving")
    box(5.3, 3.6, 3.3, 1.1, "Driver", "can brake below the cap")
    box(0.2, 0.6, 4.5, 1.1, "Vehicle", "brakes, powertrain")
    box(5.3, 0.6, 4.5, 1.1, "Crossing (controlled process)", "pedestrians, parked vehicles")
    arrow(4.2, 8.7, 4.2, 7.5, "geofence, cap value,\nparking restrictions", -1)
    arrow(5.8, 7.5, 5.8, 8.7, "override logs,\ncasualty records", 1, MUTED)
    arrow(2.6, 6.4, 1.6, 4.7, "15 km/h cap\n+ heartbeat", -1, ACCENT)
    arrow(3.0, 4.7, 4.0, 6.4, "speed, position,\nview blocked?", 1, MUTED)
    arrow(9.2, 1.7, 9.2, 6.4, "", -1, MUTED)
    ax.text(9.08, 2.6, "pedestrian\nsensor", fontsize=7.2, color=MUTED, ha="right", va="center")
    arrow(2.3, 3.6, 2.3, 1.7, "brake / torque", -1)
    arrow(6.2, 3.6, 4.0, 1.7, "brake pedal", 1)
    arrow(4.7, 1.15, 5.3, 1.15, "", 1)
    ax.text(5.0, 0.75, "motion", fontsize=7.2, ha="center", color=INK)
    fig.savefig(FIG / "fig_control.png"); plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    for f in (fig_models, fig_forest, fig_coverage, fig_impact, fig_control):
        f()
    print(sorted(p.name for p in FIG.iterdir()))
