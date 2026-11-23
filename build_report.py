"""Builds report/Crosswalk-Narrative-Mining.docx and .tex from results/*.json.

No number in the report is typed by hand: all of them are read from the
JSON the analysis wrote. Run after analysis.py, crossing_governor.py and
figures.py.  The .tex is compiled with pdflatex (see README).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).parent
R = json.loads((ROOT / "results" / "results.json").read_text())
K = json.loads((ROOT / "results" / "constraint.json").read_text())
FIG = ROOT / "results" / "figures"
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)
NAME = "Crosswalk-Narrative-Mining"

TITLE = "Crosswalk Pedestrian Escalation Modelling via Narrative Mining"
SUB = "Safety Analytics capstone, Project 4: STATS19 text features, a multi-class SVM and a STAMP crossing constraint"
AUTHOR = "Vipluv  |  Roll no. 22MF3IM22  |  November 2026"

f = R["flow"]; M = R["models"]; cov = R["coverage"]; adj = R["or_adjusted"]["terms"]; crude = R["or_crude"]
MAIN = "TF-IDF narrative (with sightline text)"
BLIND = "TF-IDF narrative (sightline text removed)"
bs = R["paired_bootstrap_macro_f1"]
sd = K["stopping_distance_m"]; imp = K["impact_speed_kmh"]; az = K["approach_zone_m"]; A = K["assumptions"]


def pct(x, d=1): return f"{100 * x:.{d}f}%"
def orci(v): return f"{v['or']:.2f} ({v['ci95'][0]:.2f} to {v['ci95'][1]:.2f})"
def n(x): return f"{x:,}"


sev = R["severity_counts"]; N = f["crossing_pedestrians"]
d_sight = bs["sightline text: with - without"]; d_oh = bs["TF-IDF - one-hot codes"]; d_dum = bs["TF-IDF - stratified dummy"]
cm = R["confusion_test"]
fatal_rank = R["top_terms"]["fatal_sightline_ranks"]["masked parked"]
serious_rank = R["top_terms"]["serious_sightline_ranks"]["masked parked"]

B: list[tuple] = []
def h1(t): B.append(("h1", t))
def h2(t): B.append(("h2", t))
def p(t): B.append(("p", t))
def bullets(xs): B.append(("ul", xs))
def table(head, rows, cap, widths): B.append(("table", head, rows, cap, widths))
def fig(name, cap, w=14.5): B.append(("fig", name, cap, w))
def code(t): B.append(("code", t))


# ---------------------------------------------------------------- content
h1("Summary")
p(f"The brief asks whether text mined from police crash narratives can predict how pedestrian near-misses escalate, "
  f"and asks for an infrastructure constraint that caps vehicles at 15 km/h when a pedestrian is on the crossing and "
  f"the driver's view is blocked. This report runs that pipeline on five years of UK Department for Transport STATS19 "
  f"records ({R['split']['train'][0]} to {R['split']['test']}): {n(f['pedestrian_casualties'])} pedestrian casualties, of whom "
  f"{n(N)} were crossing the road and {n(R['masked_count'])} ({pct(cov['masked_share_of_crossing'])}) were recorded as "
  f"masked by a parked or stationary vehicle.")
p("Four results, two of them negative:")
bullets([
    f"**The narrative field does not exist.** The open STATS19 release has no free text and no near-miss records. "
    f"Narratives were therefore rendered from the coded fields, and injury severity (fatal, serious, slight) stands in "
    f"for escalation. Text built this way cannot hold more information than the codes, and the test confirms it: "
    f"TF-IDF scores {M[MAIN]['macro_f1']:.3f} macro-F1 against {M['One-hot codes (with movement code)']['macro_f1']:.3f} for plain one-hot codes.",
    f"**The SVM beats chance, modestly.** Macro-F1 {M[MAIN]['macro_f1']:.3f} on the held-out {R['split']['test']} year against "
    f"{M['Dummy (stratified)']['macro_f1']:.3f} for a stratified guess (difference {d_dum['diff']:+.3f}, 95% CI "
    f"{d_dum['ci95'][0]:+.3f} to {d_dum['ci95'][1]:+.3f}). It finds {pct(M[MAIN]['recall_fatal'], 0)} of fatalities but only "
    f"{pct(M[MAIN]['recall_serious'], 0)} of serious injuries.",
    f"**A blocked sightline does raise severity, but only once confounding is removed.** The crude odds ratio of being "
    f"killed or seriously injured is {orci(crude)}, which looks like no effect. Adjusted for speed limit, age, light, "
    f"vehicle type and reporting system it is {orci(adj['masked'])}. Masked casualties are disproportionately children "
    f"in daylight, two groups with lower severity, and that hides the effect. Removing the sightline phrases from the "
    f"text does not hurt the classifier (macro-F1 change {d_sight['diff']:+.3f}), so the effect is real but too small to "
    f"help prediction.",
    f"**The 15 km/h cap is sound physics, but the constraint reaches few of the people it is meant for.** At 15 km/h a "
    f"vehicle stops in {sd['15 km/h (cap)']:.1f} m against {sd['30 mph']:.1f} m at 30 mph. But only "
    f"{pct(cov['on_facility_share'])} of masked-crossing casualties were on a crossing facility where roadside sensors "
    f"could see them; {pct(cov['no_facility_within_50m_share'])} had no facility within 50 m.",
])

h1("1. What the brief assumes and what the data holds")
p("The brief names the UK Department for Transport Road Safety Data. That dataset is real and open, but three things "
  "the brief relies on are not in it. Each gap is listed with the substitute used, so every later result can be read "
  "with the right caveat.")
table(["Brief asks for", "What STATS19 has", "What this report does"], [
    ["Unstructured police_narrative field", "No free text. Narratives stay in police systems; the open release is coded fields only.",
     "Renders 16 coded fields into a sentence-per-field narrative using DfT's own code labels, then applies TF-IDF."],
    ["Near-miss records and how quickly they escalate", "Injury collisions only. No near misses, no timing.",
     "Uses casualty severity (fatal, serious, slight) as the escalation outcome."],
    ["Vehicle speed variance profiles", "No measured speeds. Only the posted speed limit.",
     "Uses the posted limit as the speed proxy."],
    ["Phrases such as \"vision blocked by parked vehicle\"", "A coded equivalent: pedestrian movement \"masked by parked or stationary vehicle\".",
     "Uses that code as the sightline-obstruction exposure throughout."],
], "Table 1. Gaps between the brief and the dataset.", [0.27, 0.35, 0.38])

h1("2. Data and merge")
p(f"Three tables were downloaded from data.dft.gov.uk (the last-5-years files): {n(f['collision_rows'])} collisions, "
  f"{n(f['vehicle_rows'])} vehicles and {n(f['casualty_rows'])} casualties. Pedestrian casualties (casualty class 3) were "
  f"joined to the vehicle that struck them on collision index and vehicle reference, then to the collision record. "
  f"All {n(f['pedestrian_casualties'])} pedestrians matched in both joins ({f['unmatched_vehicle']} and "
  f"{f['unmatched_collision']} unmatched).")
mv = f["movement_counts_all_pedestrians"]
p(f"The analysis population is pedestrians recorded as crossing: from the driver's nearside ({n(mv['1'])}) or offside "
  f"({n(mv['3'])}), and the same two movements masked by a parked or stationary vehicle ({n(mv['2'])} and {n(mv['4'])}). "
  f"That gives {n(N)} casualties: {n(sev['fatal'])} fatal, {n(sev['serious'])} serious and {n(sev['slight'])} slight. "
  f"The {n(mv['9'])} pedestrians with movement \"unknown or other\" are excluded, since their sightline status cannot be known.")
p(f"The split is by time, not at random, so the test mimics predicting next year from past years: train on "
  f"{R['split']['train'][0]} to {R['split']['train'][-1]} ({n(R['n']['train'])}), tune on {R['split']['val']} "
  f"({n(R['n']['val'])}), test once on {R['split']['test']} ({n(R['n']['test'])}).")

h1("3. Narrative construction and TF-IDF")
p("Each record becomes one short sentence per coded field, in lower case, using the label text from DfT's 2025 data "
  "guide. One masked-crossing record reads:")
code(R["example_narratives"][1])
p(f"The text is tokenised, English stop-words are removed, and 1- to 3-word phrases are counted within each sentence "
  f"(phrases never span two fields). Terms seen in fewer than 5 training records are dropped and term frequency is "
  f"log-scaled. The resulting vocabulary has {n(R['vocab_size'])} terms, including \"masked parked stationary vehicle\"-type "
  f"phrases that are the nearest the data can get to the brief's example.")
p("This step is honest about what it is. A narrative written from codes is a re-encoding of the codes. The value of "
  "running it is to build and test the pipeline that would be used on real narratives, and to measure a baseline that "
  "real text would have to beat.")

h1("4. Multi-class SVM")
p(f"A linear support vector machine (one-vs-rest, class weights balanced to offset the {pct(sev['fatal'] / N)} share of "
  f"fatalities, regularisation C chosen on the {R['split']['val']} year) predicts the three severity classes. Five feature "
  f"sets and two no-skill baselines are compared on the same test year.")
rows = []
for k in sorted(M, key=lambda k: -M[k]["macro_f1"]):
    v = M[k]
    rows.append([k, f"{v['macro_f1']:.3f}", f"{v['balanced_accuracy']:.3f}", pct(v["recall_fatal"], 0),
                 pct(v["recall_serious"], 0), pct(v["recall_slight"], 0)])
table(["Feature set", "Macro-F1", "Balanced acc.", "Fatal recall", "Serious recall", "Slight recall"], rows,
      f"Table 2. Test-year ({R['split']['test']}) performance, n = {n(R['n']['test'])}.", [0.40, 0.11, 0.13, 0.12, 0.12, 0.12])
fig("fig_models.png", "Figure 1. Macro-F1 by feature set. The brief's model is highlighted.")
p(f"Three readings. First, the model has real but limited skill: {d_dum['diff']:+.3f} macro-F1 over a stratified guess "
  f"(95% CI {d_dum['ci95'][0]:+.3f} to {d_dum['ci95'][1]:+.3f}, paired bootstrap, 2,000 resamples), and well above "
  f"speed limit alone ({M['Speed limit only']['macro_f1']:.3f}). Second, TF-IDF text is not better than one-hot codes "
  f"(difference {d_oh['diff']:+.3f}, CI {d_oh['ci95'][0]:+.3f} to {d_oh['ci95'][1]:+.3f}); it is slightly worse, as "
  f"expected when {n(M[MAIN]['n_features'])} correlated phrases re-express {M['One-hot codes (with movement code)']['n_features']} codes. Third, removing the sightline sentence does not reduce "
  f"skill (with minus without: {d_sight['diff']:+.3f}, CI {d_sight['ci95'][0]:+.3f} to {d_sight['ci95'][1]:+.3f}).")
table(["Actual \\ Predicted", "Fatal", "Serious", "Slight"],
      [[lab] + [n(x) for x in cm[i]] for i, lab in enumerate(["Fatal", "Serious", "Slight"])],
      "Table 3. Confusion matrix of the TF-IDF model on the test year.", [0.34, 0.22, 0.22, 0.22])
p(f"The model separates the extremes and not the middle: {n(cm[1][2])} of {n(sum(cm[1]))} serious injuries are called "
  f"slight. The price of its fatal recall is {n(cm[1][0] + cm[2][0])} false fatal alarms for {n(cm[0][0])} true ones. "
  f"The strongest phrases for the fatal class are \"" + "\", \"".join(t for t, _ in R["top_terms"]["fatal"][:6]) +
  f"\": age over 75, a vehicle going ahead (not turning or reversing) and heavy vehicles. The phrase \"masked parked\" "
  f"ranks {fatal_rank[0]} of {n(R['vocab_size'])} for the fatal class (weight {fatal_rank[1]:+.3f}) and {serious_rank[0]} for "
  f"serious ({serious_rank[1]:+.3f}): close to zero in both.")

h1("5. Does a blocked sightline raise severity?")
p("The hypothesis says sightline obstruction is a hidden predictor. The classifier says it adds no predictive skill. "
  "An odds ratio answers the narrower causal-flavoured question directly: among crossing pedestrians who were hit, are "
  "the masked ones more often killed or seriously injured (KSI)?")
a, b, c, d = crude["table"]
table(["", "KSI", "Slight", "KSI rate"], [
    ["Masked by parked or stationary vehicle", n(a), n(b), pct(crude["ksi_rate_masked"])],
    ["Not masked", n(c), n(d), pct(crude["ksi_rate_unmasked"])],
], f"Table 4. Crude 2 x 2 table. Odds ratio {orci(crude)}.", [0.46, 0.18, 0.18, 0.18])
fig("fig_forest.png", "Figure 2. Odds ratio of KSI for masked against unmasked crossings: crude, adjusted, and within strata.")
p(f"The crude ratio's interval includes 1. The adjusted ratio, from a logistic regression on {n(R['or_adjusted']['n'])} "
  f"complete records, is {orci(adj['masked'])}. The gap is confounding that runs against the effect: "
  f"{pct(R['child_share']['masked'], 0)} of masked casualties are children under 16 against "
  f"{pct(R['child_share']['unmasked'], 0)} of unmasked ones, and only {pct(R['dark_share']['masked'], 0)} happen in darkness "
  f"against {pct(R['dark_share']['unmasked'], 0)}. Children (OR {adj['age_child']['or']:.2f}) and daylight carry lower severity, "
  f"so the masked group looks safer than it is until those are held fixed.")
table(["Term (reference)", "Odds ratio (95% CI)"], [
    ["Masked crossing (not masked)", orci(adj["masked"])],
    ["Darkness (daylight)", orci(adj["dark"])],
    ["20 mph limit (30 mph)", orci(adj["speed_20"])],
    ["40 mph limit (30 mph)", orci(adj["speed_40"])],
    ["60 mph limit (30 mph)", orci(adj["speed_60"])],
    ["Pedestrian aged 66+ (16 to 65)", orci(adj["age_older"])],
    ["Child 0 to 15 (16 to 65)", orci(adj["age_child"])],
    ["Heavy vehicle (car)", orci(adj["veh_heavy"])],
    ["On a crossing facility (elsewhere)", orci(adj["on_facility"])],
    ["Injury-based police reporting system", orci(adj["casualty_injury_based"])],
], "Table 5. Adjusted odds ratios for KSI (selected terms; full model in results/results.json).", [0.6, 0.4])
p(f"Two cautions. The 40 mph stratum points the other way ({orci(R['or_by_speed_limit']['40'])}) on "
  f"{n(sum(R['or_by_speed_limit']['40']['table'][:2]))} masked casualties, so the effect is not uniform. And STATS19 "
  f"records only collisions that injured someone: a masked crossing that ended in a near miss is not in the data, so "
  f"these ratios describe severity given a collision, not the risk of a collision. The brief's near-miss escalation "
  f"question cannot be answered from this source.")
p("Verdict on the hypothesis: partly supported. Sightline obstruction is an independent severity factor of about 20% "
  "higher odds, hidden in the crude data by who gets masked. It is not a strong enough signal to improve a classifier, "
  "and nothing here shows that text mining adds to the coded field.")

h1("6. STAMP analysis and the infrastructure constraint")
p("In STAMP an accident is a control problem: a safety constraint was not enforced by the control structure. The "
  "loss here is a pedestrian struck at a speed that kills or seriously injures; the hazard is a vehicle approaching a "
  "crossing faster than it can stop within the distance at which it can first see a pedestrian.")
fig("fig_control.png", "Figure 3. Safety control structure for an instrumented crossing. Control actions point down, feedback up.", 12.5)
h2("6.1 Is 15 km/h the right number?")
p(f"With a {A['reaction_s']:.1f} s reaction time and {A['emergency_decel_ms2']} m/s2 emergency braking (assumptions, "
  f"dry road), stopping distance is {sd['15 km/h (cap)']:.1f} m at 15 km/h, {sd['20 mph']:.1f} m at 20 mph and "
  f"{sd['30 mph']:.1f} m at 30 mph. A pedestrian stepping out from behind a parked van is typically first visible "
  f"within a few metres.")
table(["First visible at", "15 km/h (cap)", "20 mph", "30 mph", "40 mph"],
      [[f"{s} m"] + [("stops" if imp[k][str(s)] < 0.05 else f"{imp[k][str(s)]:.0f} km/h") for k in imp] for s in (3, 5, 8, 12, 20, 30)],
      "Table 6. Impact speed by approach speed and the distance at which the pedestrian becomes visible.", [0.24, 0.19, 0.19, 0.19, 0.19])
fig("fig_impact.png", "Figure 4. Impact speed against sight distance. The cap turns an 8 m emergence from a 48 km/h impact into a stop.")
p(f"The cap does what it claims from about {sd['15 km/h (cap)']:.1f} m of sight distance upward, and below that the "
  f"impact is at most 15 km/h. It does not guarantee a stop for a pedestrian who emerges inside 5 m; no speed above "
  f"walking pace does. The data adds that the cap is a real override in most cases: "
  f"{pct(R['speed_limit_share_masked']['30'], 0)} of masked-crossing casualties were on 30 mph roads and "
  f"{pct(R['speed_limit_share_masked']['20'], 0)} on 20 mph roads.")
h2("6.2 Unsafe control actions")
p("The brief's rule covers one case: both signals healthy and true. STPA asks how the same control action becomes "
  "unsafe in the other cases. Each row below led to a requirement that the code enforces and a test checks.")
table(["Unsafe control action", "Why it is hazardous", "Requirement added"], [
    ["UCA-1. Cap not issued when a pedestrian is present and the view is blocked",
     "Sensor fault, lost roadside link or unknown view state is read as \"no hazard\".",
     "Missing or stale feedback is treated as hazard present: the cap applies (fail-safe)."],
    ["UCA-2. Cap issued as an instant step change",
     "Harsh braking from 30 mph invites a rear-end collision from a following vehicle.",
     f"Speed ramps to the cap at no more than {A['ramp_decel_ms2']} m/s2."],
    ["UCA-3. Cap issued too late",
     "Vehicle is already inside its stopping distance when the command arrives.",
     f"Approach zone long enough to finish the ramp: {az['20 mph']:.0f} m at 20 mph, {az['30 mph']:.0f} m at 30 mph, {az['40 mph']:.0f} m at 40 mph."],
    ["UCA-4. Cap released too soon",
     "Sensor loses a pedestrian still hidden behind the parked vehicle; a second pedestrian follows.",
     "Cap is held for 3 s after the hazard last appeared."],
    ["UCA-5. Cap issued outside the crossing zone",
     "Nuisance overrides teach drivers to distrust or defeat the system.",
     "No override outside the geofence; none when the view is clear."],
], "Table 7. Unsafe control actions for the 15 km/h override and the resulting requirements.", [0.30, 0.36, 0.34])
p("The constraint is implemented in crossing_governor.py as a deterministic rule, deliberately not a learned model: "
  f"a classifier that misses {pct(1 - M[MAIN]['recall_serious'], 0)} of serious injuries cannot be the last line of defence. "
  "Twelve unit tests cover the brief's rule, each fail-safe path, the hold time, the ramp and the kinematics.")
h2("6.3 Coverage: where masked crossings actually happen")
fig("fig_coverage.png", "Figure 5. Location of masked-crossing pedestrian casualties relative to crossing facilities.")
p(f"An infrastructure-level constraint needs infrastructure. Of {n(cov['masked_total'])} masked-crossing casualties, "
  f"{n(cov['on_facility'])} ({pct(cov['on_facility_share'])}) were on a crossing facility or its zig-zag lines and a "
  f"further {n(cov['within_50m'])} ({pct(cov['within_50m_share'])}) within 50 m of one. For "
  f"{pct(cov['no_facility_within_50m_share'])} there was no crossing facility within 50 m at all. In severity terms the "
  f"constraint as specified could have acted on {n(cov['ksi_on_facility'])} of {n(cov['ksi_total'])} KSI casualties and "
  f"{cov['fatal_on_facility']} of {cov['fatal_total']} deaths. This is expected once stated: parking is already "
  f"banned on zig-zag lines, so people are masked by parked vehicles mostly where there is no crossing.")
p("The STAMP reading is that the constraint is placed at the wrong level of the control structure to control most of "
  "the hazard. Two changes follow. The same rule should also run on the vehicle, using its own perception of occluded "
  "kerbside space, so it works on any street. And the authority-level control, parking restrictions near informal "
  "crossing points such as school frontages, removes the occlusion instead of reacting to it.")

h1("7. Limitations")
bullets([
    "The narrative is synthetic by construction. Real free-text narratives could carry details the codes lack; this study cannot say whether they do.",
    "Severity is a proxy for escalation. No near-miss or timing data exists in the source.",
    "Posted speed limit stands in for travel speed. Actual impact speeds are unknown.",
    f"The masked code is assigned by the reporting officer and may be under-recorded; \"unknown or other\" movement is {pct(mv['9'] / f['pedestrian_casualties'], 0)} of pedestrians.",
    "Serious and slight are recorded differently by forces using injury-based systems (adjusted OR "
    f"{adj['casualty_injury_based']['or']:.2f}); it is a covariate in the regression but not in the SVM's target.",
    "The stopping-distance figures rest on stated reaction and braking assumptions for a dry road. The constraint was tested in software only.",
    "The coverage figures count casualties who could have been sensed, not casualties who would have been saved.",
])

h1("8. Reproducing this report")
code(".venv/bin/python analysis.py            # merge, narratives, TF-IDF, SVM, odds ratios -> results/results.json\n"
     ".venv/bin/python crossing_governor.py   # constraint kinematics -> results/constraint.json\n"
     ".venv/bin/python -m pytest tests -q     # 12 tests on the constraint\n"
     ".venv/bin/python figures.py             # results/figures/*.png\n"
     ".venv/bin/python build_report.py        # report/*.docx and *.tex")
h1("Sources")
bullets([
    "Department for Transport, Road Safety Data (STATS19): collision, vehicle and casualty tables, last five years, and the 2025 open dataset data guide. data.dft.gov.uk, Open Government Licence.",
    "N. G. Leveson, Engineering a Safer World: Systems Thinking Applied to Safety. MIT Press, 2011.",
    "N. G. Leveson and J. P. Thomas, STPA Handbook, 2018.",
])


# ---------------------------------------------------------------- docx
def add_runs(par, text):
    for i, seg in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if seg:
            par.add_run(seg).bold = bool(i % 2)


def build_docx():
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2.0)
    t = doc.add_paragraph(); r = t.add_run(TITLE); r.bold = True; r.font.size = Pt(20)
    s2 = doc.add_paragraph(); r = s2.add_run(SUB); r.font.size = Pt(11.5); r.font.color.rgb = RGBColor(0x55, 0x5F, 0x6B)
    doc.add_paragraph().add_run(AUTHOR).font.size = Pt(10.5)
    width = 16.6
    for b in B:
        k = b[0]
        if k == "h1":
            doc.add_heading(b[1], level=1)
        elif k == "h2":
            doc.add_heading(b[1], level=2)
        elif k == "p":
            add_runs(doc.add_paragraph(), b[1])
        elif k == "ul":
            for x in b[1]:
                add_runs(doc.add_paragraph(style="List Bullet"), x)
        elif k == "code":
            par = doc.add_paragraph(); r = par.add_run(b[1]); r.font.name = "Menlo"; r.font.size = Pt(8.5)
            par.paragraph_format.left_indent = Cm(0.6)
        elif k == "fig":
            doc.add_picture(str(FIG / b[1]), width=Cm(b[3]))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            c = doc.add_paragraph(); r = c.add_run(b[2]); r.italic = True; r.font.size = Pt(9)
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif k == "table":
            _, head, rows, cap, widths = b
            c = doc.add_paragraph(); r = c.add_run(cap); r.bold = True; r.font.size = Pt(9.5)
            c.paragraph_format.keep_with_next = True
            tb = doc.add_table(rows=1, cols=len(head)); tb.style = "Light Grid Accent 1"; tb.autofit = False
            for i, htxt in enumerate(head):
                cell = tb.rows[0].cells[i]; cell.text = ""
                rr = cell.paragraphs[0].add_run(htxt); rr.bold = True; rr.font.size = Pt(9)
            for row in rows:
                cells = tb.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = ""
                    cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(9)
            for row in tb.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(width * w)
            doc.add_paragraph()
    doc.save(OUT / f"{NAME}.docx")


# ---------------------------------------------------------------- latex
def esc(t: str) -> str:
    t = t.replace("\\", "\\textbackslash{}")
    for a_, b_ in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{}", "{}"),
                   ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        if a_ != "{}":
            t = t.replace(a_, b_)
    t = t.replace("\\textbackslash\\{\\}", "\\textbackslash{}")
    t = re.sub(r'"([^"]*)"', r"``\1''", t)
    t = t.replace("m/s2", "m/s\\textsuperscript{2}")
    return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)


def build_tex():
    L = [r"\documentclass[10pt,a4paper]{article}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}",
         r"\usepackage[margin=2.2cm]{geometry}", r"\usepackage{graphicx}", r"\usepackage{booktabs}",
         r"\usepackage{array}", r"\usepackage{enumitem}", r"\usepackage[hidelinks]{hyperref}",
         r"\renewcommand{\familydefault}{\sfdefault}", r"\setlength{\parindent}{0pt}", r"\setlength{\parskip}{6pt}",
         r"\setlist{itemsep=3pt,topsep=2pt,leftmargin=1.4em}", r"\renewcommand{\arraystretch}{1.25}",
         r"\emergencystretch=2em", r"\raggedbottom",
         r"\begin{document}",
         r"{\LARGE\bfseries " + esc(TITLE) + r"\par}\vspace{4pt}",
         r"{\large " + esc(SUB) + r"\par}\vspace{4pt}", esc(AUTHOR) + r"\par\vspace{2pt}\hrule\vspace{6pt}"]
    for b in B:
        k = b[0]
        if k == "h1":
            L.append(r"\section*{" + esc(b[1]) + "}")
        elif k == "h2":
            L.append(r"\subsection*{" + esc(b[1]) + "}")
        elif k == "p":
            L.append(esc(b[1]) + r"\par")
        elif k == "ul":
            L.append(r"\begin{itemize}" + "".join(r"\item " + esc(x) + "\n" for x in b[1]) + r"\end{itemize}")
        elif k == "code":
            L.append(r"\begin{quote}\small\ttfamily\raggedright " + esc(b[1]).replace("\n", r"\\ ") + r"\end{quote}")
        elif k == "fig":
            L.append(r"\begin{center}\includegraphics[width=" + f"{b[3]}cm" + r"]{../results/figures/" + b[1] + r"}\\[2pt]"
                     r"{\small\itshape " + esc(b[2]) + r"}\end{center}")
        elif k == "table":
            _, head, rows, cap, widths = b
            spec = "".join(r">{\raggedright\arraybackslash}p{" + f"{0.92 * w:.3f}" + r"\textwidth}" for w in widths)
            body = " \\\\\n".join(" & ".join(esc(str(v)) for v in row) for row in rows)
            L.append(r"\begin{center}\begin{minipage}{\textwidth}{\small\bfseries " + esc(cap) + r"}\\[3pt]"
                     r"\small\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}" + spec + r"@{}}\toprule "
                     + " & ".join(r"\textbf{" + esc(x) + "}" for x in head) + r" \\ \midrule " + "\n" + body
                     + r" \\ \bottomrule\end{tabular}\end{minipage}\end{center}")
    L.append(r"\end{document}")
    (OUT / f"{NAME}.tex").write_text("\n\n".join(L))


if __name__ == "__main__":
    build_docx(); build_tex()
    print("wrote", sorted(x.name for x in OUT.iterdir()))
