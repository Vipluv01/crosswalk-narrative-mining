# Crosswalk pedestrian escalation modelling (Safety Analytics, Project 4)

Runs the brief's four-step pipeline on five years of UK DfT STATS19 road-safety data:
merge the Collision, Vehicle and Casualty tables for crossing pedestrians, turn each
record into a narrative and TF-IDF it, train a multi-class linear SVM on injury severity,
and design and test a STAMP crossing constraint (15 km/h cap when a pedestrian is active
and the forward view is blocked).

Self-contained: its own `.venv`, unrelated to the other folders next to it.

## The report

`report/Crosswalk-Narrative-Mining.pdf` and `.docx` (same content), plus an 8-slide
`.pptx` built from the same results. Every number in it is
read from `results/results.json` and `results/constraint.json`; none is typed by hand.

## What to know before reading it

The open STATS19 release has **no `police_narrative` field, no near-miss records and no
measured speeds**. Narratives are rendered from the coded fields with DfT's own labels,
injury severity stands in for escalation, and the posted limit stands in for speed. The
report says so up front and measures what that costs. Two of its four headline results
are negative; they are findings, not bugs.

## Reproduce

```bash
# data (about 255 MB, not kept in git): the three last-5-years CSVs and the 2025 data guide
B=https://data.dft.gov.uk/road-accidents-safety-data
for t in collision vehicle casualty; do curl -o data/$t-last-5-years.csv $B/dft-road-casualty-statistics-$t-last-5-years.csv; done
# data guide: linked from https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data
#   save as data/data-guide-2025.xlsx

.venv/bin/python analysis.py            # about 80 s -> results/results.json
.venv/bin/python crossing_governor.py   # -> results/constraint.json
.venv/bin/python -m pytest tests -q     # 12 tests on the constraint
.venv/bin/python figures.py             # -> results/figures/
.venv/bin/python build_report.py        # -> report/*.docx, *.tex
.venv/bin/python build_slides.py        # -> report/*.pptx, then audits every shape
cd report && pdflatex Crosswalk-Narrative-Mining.tex   # TinyTeX, twice
```

Environment: Python 3.11, numpy, pandas, scikit-learn 1.9.1, scipy, statsmodels,
matplotlib, openpyxl, python-docx, python-pptx, pytest (`requirements.txt` has the exact pins).

## Files

| File | What it is |
|---|---|
| `analysis.py` | merge, narratives, TF-IDF, SVM and ablations, odds ratios, coverage |
| `crossing_governor.py` | the STAMP constraint as code, plus stopping-distance kinematics |
| `tests/test_governor.py` | the brief's rule, fail-safe paths, hold time, ramp |
| `figures.py`, `build_report.py` | figures and the Word/LaTeX report, from the JSON |
| `build_slides.py` | the 8-slide deck, from the same JSON, with a geometry audit |
| `results/run_output.txt` | console output of the analysis run the report is built from |
