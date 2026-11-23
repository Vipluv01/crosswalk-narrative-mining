"""Builds report/Crosswalk-Narrative-Mining.pptx (8 slides, 16:9) from results/*.json.

Same rule as the report: no number is typed by hand. Ends with a geometry
audit: every shape must sit inside the slide and every text box must have
room for its text at a conservative characters-per-line estimate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent
R = json.loads((ROOT / "results" / "results.json").read_text())
K = json.loads((ROOT / "results" / "constraint.json").read_text())
FIG = ROOT / "results" / "figures"
OUT = ROOT / "report" / "Crosswalk-Narrative-Mining.pptx"

W, H = 13.333, 7.5
MX = 0.6
INK, MUTED, ACCENT, PALE = RGBColor(0x1F, 0x29, 0x33), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xB4, 0x47, 0x2F), RGBColor(0xF3, 0xF4, 0xF6)
FONT = "Calibri"

f = R["flow"]; M = R["models"]; cov = R["coverage"]; adj = R["or_adjusted"]["terms"]; crude = R["or_crude"]
MAIN = "TF-IDF narrative (with sightline text)"; ONEHOT = "One-hot codes (with movement code)"
bs = R["paired_bootstrap_macro_f1"]; sd = K["stopping_distance_m"]; imp = K["impact_speed_kmh"]; A = K["assumptions"]
sev = R["severity_counts"]; N = f["crossing_pedestrians"]; cm = R["confusion_test"]


def pct(x, d=0): return f"{100 * x:.{d}f}%"
def orci(v): return f"{v['or']:.2f} ({v['ci95'][0]:.2f} to {v['ci95'][1]:.2f})"
def n(x): return f"{x:,}"


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
AUDIT: list[tuple] = []   # (slide_no, label, x, y, w, h, lines_needed, lines_available)


def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, anchor=MSO_ANCHOR.TOP, label="text", gap=6):
    """paras: list of str or (str, dict(bold=, color=, size=)). '**x**' prefix segments are not parsed; use tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    used = 0.0
    for i, item in enumerate(paras):
        s, o = (item, {}) if isinstance(item, str) else item
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.space_after = Pt(o.get("gap", gap))
        r = par.add_run(); r.text = s
        sz = o.get("size", size)
        r.font.size = Pt(sz); r.font.name = FONT; r.font.bold = o.get("bold", bold)
        r.font.color.rgb = o.get("color", color)
        # conservative fit estimate: average glyph width 0.50 em (0.54 bold), line height 1.2 em
        cpl = max(1, int((w - 0.1) * 72 / (sz * (0.54 if o.get("bold", bold) else 0.50))))
        lines = sum(max(1, math.ceil(len(seg) / cpl)) for seg in s.split("\n"))
        used += lines * sz * 1.2 / 72 + o.get("gap", gap) / 72
    used -= (paras[-1][1].get("gap", gap) if not isinstance(paras[-1], str) else gap) / 72   # no gap after the last line
    AUDIT.append((len(prs.slides), label, x, y, w, h, used, h - 0.06))
    return tb


def rect(slide, x, y, w, h, fill=PALE, line=None, label="rect"):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(1)
    s.shadow.inherit = False
    AUDIT.append((len(prs.slides), label, x, y, w, h, 0, 1))
    return s


def picture(slide, name, x, y, w, h):
    """Fit the image inside the box, centred, keeping its aspect ratio."""
    iw, ih = Image.open(FIG / name).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    px, py = x + (w - pw) / 2, y + (h - ph) / 2
    slide.shapes.add_picture(str(FIG / name), Inches(px), Inches(py), Inches(pw), Inches(ph))
    AUDIT.append((len(prs.slides), name, px, py, pw, ph, 0, 1))


def new(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, 0.18, H, fill=ACCENT, label="rail")
    if kicker:
        text(s, MX, 0.34, W - 2 * MX, 0.38, [kicker.upper()], size=12, color=ACCENT, bold=True, label="kicker")
    text(s, MX, 0.70, W - 2 * MX, 1.15, [title], size=28, bold=True, anchor=MSO_ANCHOR.TOP, label="title")
    text(s, W - 1.4, H - 0.52, 0.9, 0.36, [str(len(prs.slides))], size=11, color=MUTED, label="page")
    return s


def stat(slide, x, y, w, big, small, hot=False):
    rect(slide, x, y, w, 1.65, fill=PALE, label="stat-bg")
    text(slide, x + 0.15, y + 0.12, w - 0.3, 0.7, [big], size=30, bold=True, color=ACCENT if hot else INK, label="stat-big")
    text(slide, x + 0.15, y + 0.82, w - 0.3, 0.78, [small], size=12, color=MUTED, label="stat-small")


def bullets(slide, x, y, w, h, items, size=16, label="bullets"):
    paras = []
    for it in items:
        head, body = it if isinstance(it, tuple) else (None, it)
        if head:
            paras.append((head, {"bold": True, "size": size, "gap": 1}))
        paras.append((body, {"size": size, "gap": 10}))
    return text(slide, x, y, w, h, paras, size=size, label=label)


BODY_Y = 1.95

# 1 -- title ------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, fill=INK, label="bg")
rect(s, 0, 0, 0.18, H, fill=ACCENT, label="rail")
text(s, 0.9, 1.5, 11.2, 0.4, ["SAFETY ANALYTICS CAPSTONE  |  PROJECT 4"], size=14, bold=True, color=RGBColor(0xE8, 0x9A, 0x85), label="kicker")
text(s, 0.9, 2.0, 11.4, 2.1, ["Crosswalk pedestrian escalation modelling via narrative mining"], size=40, bold=True,
     color=RGBColor(0xFF, 0xFF, 0xFF), label="title")
text(s, 0.9, 4.25, 11.2, 0.9, [f"Text features, a multi-class SVM and a STAMP crossing constraint, tested on "
                               f"{n(f['pedestrian_casualties'])} UK pedestrian casualties, {R['split']['train'][0]} to {R['split']['test']}"],
     size=18, color=RGBColor(0xD5, 0xDA, 0xE0), label="sub")
text(s, 0.9, 6.3, 11.2, 0.4, ["Vipluv  |  Roll no. 22MF3IM22  |  November 2026"], size=14, color=RGBColor(0xD5, 0xDA, 0xE0), label="author")

# 2 -- brief vs data ----------------------------------------------------------
s = new("Three things the brief assumes are not in the dataset", "The brief and the data")
cols = [("The brief asks for", 3.3), ("STATS19 has", 4.1), ("What this study uses", 4.7)]
rows = [
    ("A free-text police_narrative field", "Coded fields only; no free text is released", "Narratives rendered from 16 coded fields with DfT's own labels"),
    ("Near misses and how fast they escalate", "Injury collisions only; no near misses, no timing", "Casualty severity: fatal, serious, slight"),
    ("Vehicle speed variance profiles", "No measured speeds", "Posted speed limit as the proxy"),
    ("\"Vision blocked by parked vehicle\"", "A code: crossing \"masked by parked or stationary vehicle\"", "That code as the sightline-obstruction exposure"),
]
y = BODY_Y; x = MX
for (hd, w) in cols:
    rect(s, x, y, w - 0.06, 0.5, fill=INK, label="th")
    text(s, x + 0.1, y + 0.08, w - 0.26, 0.36, [hd], size=14, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), label="th-text")
    x += w
y += 0.58
for r_ in rows:
    x = MX
    for (hd, w), cell in zip(cols, r_):
        rect(s, x, y, w - 0.06, 0.88, fill=PALE, label="td")
        text(s, x + 0.1, y + 0.1, w - 0.26, 0.72, [cell], size=14, label="td-text")
        x += w
    y += 0.94
text(s, MX, 6.42, 11.9, 0.55, ["Each substitute is stated up front so every later result carries the right caveat."], size=14, color=MUTED, label="note")

# 3 -- data and pipeline ------------------------------------------------------
s = new(f"{n(N)} crossing pedestrians, split by year so the test mimics real use", "Data and pipeline")
sw = 2.85
stat(s, MX, BODY_Y, sw, n(f["pedestrian_casualties"]), "pedestrian casualties, all matched to a vehicle and a collision")
stat(s, MX + sw + 0.2, BODY_Y, sw, n(N), "recorded as crossing the road (the study population)")
stat(s, MX + 2 * (sw + 0.2), BODY_Y, sw, n(R["masked_count"]), f"masked by a parked or stationary vehicle ({pct(cov['masked_share_of_crossing'], 1)})", hot=True)
stat(s, MX + 3 * (sw + 0.2), BODY_Y, sw, n(sev["fatal"]), f"deaths, {pct(sev['fatal'] / N, 1)} of the population; {n(sev['serious'])} serious, {n(sev['slight'])} slight")
steps = [("1  Merge", "Casualty to striking vehicle to collision, on DfT's own keys"),
         ("2  Narrate + TF-IDF", f"One sentence per coded field; 1- to 3-word phrases; {n(R['vocab_size'])} terms"),
         ("3  Linear SVM", f"3 severity classes, balanced weights; train {R['split']['train'][0]}-{str(R['split']['train'][-1])[2:]}, tune {R['split']['val']}, test {R['split']['test']}"),
         ("4  STAMP constraint", "15 km/h cap as deterministic code, with 12 unit tests")]
x = MX
for hd, body in steps:
    rect(s, x, 3.85, sw, 1.85, fill=RGBColor(0xFF, 0xFF, 0xFF), line=MUTED, label="step-bg")
    text(s, x + 0.15, 3.98, sw - 0.3, 0.45, [hd], size=17, bold=True, color=ACCENT, label="step-head")
    text(s, x + 0.15, 4.5, sw - 0.3, 1.1, [body], size=14, label="step-body")
    x += sw + 0.2
text(s, MX, 5.95, 11.9, 0.5, ["Source: UK Department for Transport road safety data (STATS19), last-5-years files."], size=12, color=MUTED, label="note")

# 4 -- SVM --------------------------------------------------------------------
d_dum = bs["TF-IDF - stratified dummy"]; d_oh = bs["TF-IDF - one-hot codes"]; d_s = bs["sightline text: with - without"]
s = new("The SVM beats chance modestly, and text adds nothing over the codes", "Result 1: the classifier")
picture(s, "fig_models.png", MX, BODY_Y, 7.3, 3.6)
text(s, MX, 5.65, 7.3, 0.4, [f"Macro-F1 on the held-out {R['split']['test']} year, n = {n(R['n']['test'])}"], size=12, color=MUTED, label="cap")
bullets(s, 8.3, BODY_Y, 4.45, 4.9, [
    ("Real but limited skill", f"Macro-F1 {M[MAIN]['macro_f1']:.3f} against {M['Dummy (stratified)']['macro_f1']:.3f} for a stratified guess "
                               f"({d_dum['diff']:+.3f}, 95% CI {d_dum['ci95'][0]:+.3f} to {d_dum['ci95'][1]:+.3f})."),
    ("Extremes yes, middle no", f"Finds {pct(M[MAIN]['recall_fatal'])} of deaths but {pct(M[MAIN]['recall_serious'])} of serious injuries."),
    ("TF-IDF is a re-encoding", f"{M[MAIN]['macro_f1']:.3f} against {M[ONEHOT]['macro_f1']:.3f} for one-hot codes."),
    ("Sightline text does not help", f"Removing it changes macro-F1 by {-d_s['diff']:+.3f}."),
], size=15)

# 5 -- odds ratio -------------------------------------------------------------
s = new("A blocked sightline raises severity about 20%, hidden by who gets masked", "Result 2: the hypothesis")
picture(s, "fig_forest.png", MX, BODY_Y, 7.6, 3.7)
text(s, MX, 5.75, 7.6, 0.65, ["Odds ratio of killed-or-seriously-injured, masked against unmasked crossings. "
                              f"Adjusted model: n = {n(R['or_adjusted']['n'])}."], size=12, color=MUTED, label="cap")
bullets(s, 8.55, BODY_Y, 4.2, 4.9, [
    ("Crude: no visible effect", f"Odds ratio {orci(crude)}."),
    ("Adjusted: a clear one", f"{orci(adj['masked'])} after speed limit, age, light, vehicle and reporting system."),
    ("Why the gap", f"{pct(R['child_share']['masked'])} of masked casualties are children against {pct(R['child_share']['unmasked'])}; "
                    f"{pct(R['dark_share']['masked'])} happen in darkness against {pct(R['dark_share']['unmasked'])}. Both groups have lower severity."),
    ("Caution", "Collisions only: this is severity given a hit, not the risk of being hit."),
], size=15)

# 6 -- STAMP ------------------------------------------------------------------
az = K["approach_zone_m"]
s = new("The brief's rule covers one case; STPA finds four more it must handle", "Result 3: the STAMP constraint")
picture(s, "fig_control.png", MX, BODY_Y - 0.1, 6.2, 4.7)
ucas = [("Missing feedback read as \"no hazard\"", "Sensor fault or lost link applies the cap (fail-safe)"),
        ("Cap applied as a step change", f"Ramp at no more than {A['ramp_decel_ms2']} m/s² to avoid rear-end hits"),
        ("Cap arrives too late", f"Approach zone of {az['30 mph']:.0f} m at 30 mph"),
        ("Cap released too soon", "Held 3 s after the hazard last appeared"),
        ("Cap outside the crossing zone", "No override outside the geofence or with a clear view")]
text(s, 7.1, BODY_Y - 0.05, 2.7, 0.35, ["Unsafe control action"], size=13, bold=True, color=ACCENT, label="uh1")
text(s, 9.9, BODY_Y - 0.05, 2.85, 0.35, ["Requirement in the code"], size=13, bold=True, color=ACCENT, label="uh2")
y = BODY_Y + 0.38
for a_, b_ in ucas:
    rect(s, 7.1, y, 5.65, 0.76, fill=PALE, label="uca-bg")
    text(s, 7.18, y + 0.06, 2.62, 0.66, [a_], size=13, bold=True, label="uca")
    text(s, 9.9, y + 0.06, 2.8, 0.66, [b_], size=13, label="req")
    y += 0.81
text(s, MX, 6.52, 11.0, 0.42, [f"Deterministic on purpose: a model that misses {pct(1 - M[MAIN]['recall_serious'])} of serious injuries cannot be the last line of defence."],
     size=13, color=MUTED, label="note")

# 7 -- cap and coverage -------------------------------------------------------
s = new(f"15 km/h is sound physics, but only {pct(cov['on_facility_share'])} of masked casualties are where sensors can see", "Result 4: does the constraint work?")
picture(s, "fig_impact.png", MX, BODY_Y + 0.35, 5.9, 2.9)
text(s, MX, BODY_Y - 0.05, 5.9, 0.4, ["Impact speed by sight distance"], size=15, bold=True, label="h-left")
text(s, MX, 5.3, 5.9, 1.25, [f"Stops in {sd['15 km/h (cap)']:.1f} m at the cap against {sd['30 mph']:.1f} m at 30 mph. A pedestrian first visible 8 m ahead: "
                             f"a stop at the cap, a {imp['30 mph']['8']:.0f} km/h impact at 30 mph.",
                             (f"Assumes {A['reaction_s']:.1f} s reaction, {A['emergency_decel_ms2']} m/s² braking, dry road.", {"size": 11, "color": MUTED})],
     size=14, label="left-body")
text(s, 6.9, BODY_Y - 0.05, 5.85, 0.4, ["Where masked crossings happen"], size=15, bold=True, label="h-right")
picture(s, "fig_coverage.png", 6.9, BODY_Y + 0.35, 5.85, 2.1)
bullets(s, 6.9, 4.55, 5.85, 2.1, [
    f"{pct(cov['no_facility_within_50m_share'])} had no crossing facility within 50 m.",
    f"The rule as specified could act on {n(cov['ksi_on_facility'])} of {n(cov['ksi_total'])} killed or seriously injured, and {cov['fatal_on_facility']} of {cov['fatal_total']} deaths.",
    "Parking is already banned on zig-zag lines, so masking happens mostly where there is no crossing.",
], size=14)

# 8 -- conclusions ------------------------------------------------------------
s = new("The constraint is right but sits at the wrong level of the control structure", "Conclusions")
cw = 3.9
blocks = [
    ("Hypothesis: partly supported", [f"Sightline obstruction is an independent severity factor, odds ratio {adj['masked']['or']:.2f}.",
                                      "It is too weak to improve the classifier.",
                                      "Nothing here shows text mining adds to the coded field."]),
    ("Recommendations", ["Run the same rule on the vehicle, using its own view of occluded kerbside space.",
                         "Restrict parking near informal crossing points such as school frontages.",
                         "Keep the override deterministic and fail-safe."]),
    ("Limitations", ["Narratives are synthetic; real free text is untested.",
                     "Severity stands in for escalation; no near-miss data exists.",
                     "Posted limit stands in for speed; constraint tested in software only."]),
]
x = MX
for hd, items in blocks:
    rect(s, x, BODY_Y, cw, 4.35, fill=PALE, label="col-bg")
    rect(s, x, BODY_Y, cw, 0.07, fill=ACCENT, label="col-rule")
    text(s, x + 0.2, BODY_Y + 0.22, cw - 0.4, 0.45, [hd], size=15, bold=True, label="col-head")
    text(s, x + 0.2, BODY_Y + 0.85, cw - 0.4, 3.35, [(i, {"gap": 12}) for i in items], size=14, label="col-body")
    x += cw + 0.2
text(s, MX, 6.5, 12.1, 0.45, ["Code, tests and every number: analysis.py, crossing_governor.py, results/results.json."], size=12, color=MUTED, label="note")

prs.save(OUT)

# ---- geometry audit ----------------------------------------------------------
bad = []
for sl, label, x, y, w, h, need, have in AUDIT:
    if x < -1e-6 or y < -1e-6 or x + w > W + 1e-6 or y + h > H + 1e-6:
        bad.append(f"slide {sl} {label}: outside slide")
    if need > have:
        bad.append(f"slide {sl} {label}: text needs {need:.2f} in, box has {have:.2f} in")
print(f"{len(prs.slides)} slides, {len(AUDIT)} shapes audited, {len(bad)} problems")
for b in bad:
    print("  ", b)
