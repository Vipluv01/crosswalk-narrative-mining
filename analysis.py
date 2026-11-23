"""Project 4 -- crosswalk pedestrian escalation modelling on STATS19.

Pipeline (the brief's four steps, on the data that actually exists):
  1. merge Collision + Vehicle + Casualty for crossing pedestrians
  2. render each record's coded fields into a narrative, TF-IDF it
  3. multi-class linear SVM on injury severity, with ablations
  4. evidence for the infrastructure constraint (odds ratios, coverage)

Every number the report quotes is written to results/results.json.
Run:  .venv/bin/python analysis.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score)
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 0
TRAIN_YEARS, VAL_YEAR, TEST_YEAR = (2021, 2022, 2023), 2024, 2025
SEV = {1: "fatal", 2: "serious", 3: "slight"}

# coded field -> (table in the DfT guide, sentence template)
FIELDS = {
    "age_band_of_casualty": ("casualty", "pedestrian aged {}"),
    "sex_of_casualty": ("casualty", "{} pedestrian"),
    "pedestrian_location": ("casualty", "pedestrian location {}"),
    "pedestrian_movement": ("casualty", "pedestrian movement {}"),
    "vehicle_type": ("vehicle", "struck by {}"),
    "vehicle_manoeuvre": ("vehicle", "vehicle {}"),
    "first_point_of_impact": ("vehicle", "first point of impact {}"),
    "speed_limit": ("collision", "speed limit {} mph"),
    "road_type": ("collision", "road type {}"),
    "junction_detail": ("collision", "junction {}"),
    "pedestrian_crossing": ("collision", "crossing facility {}"),
    "light_conditions": ("collision", "light {}"),
    "weather_conditions": ("collision", "weather {}"),
    "road_surface_conditions": ("collision", "road surface {}"),
    "urban_or_rural_area": ("collision", "{} area"),
    "carriageway_hazards": ("collision", "carriageway hazard {}"),
}
VISIBILITY_FIELD = "pedestrian_movement"
MASKED, UNMASKED = (2, 4), (1, 3)
TOKEN = re.compile(r"[a-z0-9]+")


def load_labels() -> dict[str, dict[int, str]]:
    g = pd.read_excel(DATA / "data-guide-2025.xlsx", sheet_name="2024_code_list")
    g.columns = ["table", "field", "code", "label", "note"]
    out: dict[str, dict[int, str]] = {}
    for f, (table, _) in FIELDS.items():
        rows = g[(g.field == f) & (g.table.str.lower() == table)]
        m = {}
        for code, label in zip(rows.code, rows.label):
            try:
                m[int(code)] = "none" if pd.isna(label) else re.sub(r"\s+", " ", str(label)).strip()
            except (TypeError, ValueError):
                pass
        out[f] = m
    return out


def load_merged() -> tuple[pd.DataFrame, dict]:
    cas = pd.read_csv(DATA / "casualty-last-5-years.csv", low_memory=False)
    veh = pd.read_csv(DATA / "vehicle-last-5-years.csv", low_memory=False)
    col = pd.read_csv(DATA / "collision-last-5-years.csv", low_memory=False)
    ped = cas[cas.casualty_class == 3]
    flow = {"casualty_rows": len(cas), "vehicle_rows": len(veh),
            "collision_rows": len(col), "pedestrian_casualties": len(ped)}
    m = ped.merge(veh, on=["collision_index", "vehicle_reference"], how="left",
                  suffixes=("", "_veh"), validate="many_to_one", indicator="_v")
    flow["unmatched_vehicle"] = int((m._v != "both").sum())
    m = m.merge(col, on="collision_index", how="left", suffixes=("", "_col"),
                validate="many_to_one", indicator="_c")
    flow["unmatched_collision"] = int((m._c != "both").sum())
    m = m[(m._v == "both") & (m._c == "both")]
    flow["movement_counts_all_pedestrians"] = {
        str(k): int(v) for k, v in m.pedestrian_movement.value_counts().sort_index().items()}
    m = m[m.pedestrian_movement.isin(MASKED + UNMASKED)].copy()
    flow["crossing_pedestrians"] = len(m)
    m["masked"] = m.pedestrian_movement.isin(MASKED).astype(int)
    m["ksi"] = (m.casualty_severity <= 2).astype(int)
    m["year"] = m.collision_year
    return m.reset_index(drop=True), flow


def narrate(df: pd.DataFrame, labels, drop: tuple[str, ...] = ()) -> list[str]:
    """Deterministically render coded fields as a police-style narrative."""
    parts = []
    for f, (_, tmpl) in FIELDS.items():
        if f in drop:
            continue
        if f == "speed_limit":
            txt = df[f].map(lambda v: tmpl.format(int(v)) if v > 0 else "speed limit unknown")
        else:
            lab = labels[f]
            txt = df[f].map(lambda v, lab=lab, tmpl=tmpl: tmpl.format(lab.get(int(v), "unknown")))
        parts.append(txt)
    return pd.concat(parts, axis=1).agg(". ".join, axis=1).str.lower().tolist()


def sentence_ngrams(doc: str) -> list[str]:
    """Stop-word-filtered 1-3-grams that never span two sentences."""
    out = []
    for sent in doc.split(". "):
        toks = [t for t in TOKEN.findall(sent) if t not in ENGLISH_STOP_WORDS]
        for n in (1, 2, 3):
            out += [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return out


def scores(y, p) -> dict:
    return {
        "macro_f1": float(f1_score(y, p, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "recall_fatal": float(recall_score(y, p, labels=[1], average="macro", zero_division=0)),
        "recall_serious": float(recall_score(y, p, labels=[2], average="macro", zero_division=0)),
        "recall_slight": float(recall_score(y, p, labels=[3], average="macro", zero_division=0)),
        "accuracy": float((np.asarray(y) == np.asarray(p)).mean()),
    }


def fit_svm(Xtr, ytr, Xva, yva):
    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LinearSVC(C=C, class_weight="balanced", random_state=SEED, max_iter=5000)
        clf.fit(Xtr, ytr)
        f = f1_score(yva, clf.predict(Xva), average="macro")
        if best is None or f > best[0]:
            best = (f, C, clf)
    return best


def boot_diff(y, pa, pb, n=2000) -> dict:
    """Paired bootstrap of macro-F1(a) - macro-F1(b) on the test set."""
    rng = np.random.default_rng(SEED)
    y, pa, pb = map(np.asarray, (y, pa, pb))
    d = np.empty(n)
    for i in range(n):
        ix = rng.integers(0, len(y), len(y))
        d[i] = (f1_score(y[ix], pa[ix], average="macro")
                - f1_score(y[ix], pb[ix], average="macro"))
    return {"diff": float(f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def odds_ratio(df) -> dict:
    a = int(((df.masked == 1) & (df.ksi == 1)).sum()); b = int(((df.masked == 1) & (df.ksi == 0)).sum())
    c = int(((df.masked == 0) & (df.ksi == 1)).sum()); d = int(((df.masked == 0) & (df.ksi == 0)).sum())
    if min(a, b, c, d) == 0:
        return {"table": [a, b, c, d], "or": None, "ci95": None}
    lor = np.log(a * d / (b * c)); se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {"table": [a, b, c, d], "or": float(np.exp(lor)),
            "ci95": [float(np.exp(lor - 1.96 * se)), float(np.exp(lor + 1.96 * se))],
            "ksi_rate_masked": a / (a + b), "ksi_rate_unmasked": c / (c + d)}


def main() -> None:
    labels = load_labels()
    df, flow = load_merged()
    res: dict = {"flow": flow, "split": {"train": list(TRAIN_YEARS), "val": VAL_YEAR, "test": TEST_YEAR}}

    res["missing_share"] = {f: float((df[f] == -1).mean()) for f in FIELDS}
    res["severity_counts"] = {SEV[k]: int(v) for k, v in df.casualty_severity.value_counts().sort_index().items()}
    res["masked_count"] = int(df.masked.sum())

    # ---- step 2: narrative + TF-IDF ---------------------------------------
    df["narrative"] = narrate(df, labels)
    df["narrative_blind"] = narrate(df, labels, drop=(VISIBILITY_FIELD,))
    res["example_narratives"] = df.loc[df.masked == 1, "narrative"].head(2).tolist()

    tr, va, te = df.year.isin(TRAIN_YEARS), df.year == VAL_YEAR, df.year == TEST_YEAR
    y = df.casualty_severity.to_numpy()
    res["n"] = {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}

    def tfidf(colname):
        v = TfidfVectorizer(analyzer=sentence_ngrams, min_df=5, sublinear_tf=True)
        return v, v.fit_transform(df.loc[tr, colname]), v.transform(df.loc[va, colname]), v.transform(df.loc[te, colname])

    def onehot(cols):
        e = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
        return e, e.fit_transform(df.loc[tr, cols]), e.transform(df.loc[va, cols]), e.transform(df.loc[te, cols])

    all_codes = list(FIELDS)
    blind_codes = [f for f in all_codes if f != VISIBILITY_FIELD]
    feats = {
        "TF-IDF narrative (with sightline text)": tfidf("narrative"),
        "TF-IDF narrative (sightline text removed)": tfidf("narrative_blind"),
        "One-hot codes (with movement code)": onehot(all_codes),
        "One-hot codes (movement code removed)": onehot(blind_codes),
        "Speed limit only": onehot(["speed_limit"]),
    }

    # ---- step 3: SVM -------------------------------------------------------
    models, preds = {}, {}
    for name, (vec, Xtr, Xva, Xte) in feats.items():
        f_val, C, clf = fit_svm(Xtr, y[tr], Xva, y[va])
        p = clf.predict(Xte)
        preds[name] = p
        models[name] = {"C": C, "n_features": int(Xtr.shape[1]), "val_macro_f1": float(f_val), **scores(y[te], p)}
        if name.startswith("TF-IDF narrative (with"):
            main_clf, main_vec = clf, vec
            res["confusion_test"] = confusion_matrix(y[te], p, labels=[1, 2, 3]).tolist()
    for strat in ("most_frequent", "stratified"):
        d = DummyClassifier(strategy=strat, random_state=SEED).fit(np.zeros((tr.sum(), 1)), y[tr])
        p = d.predict(np.zeros((te.sum(), 1)))
        preds[f"Dummy ({strat})"] = p
        models[f"Dummy ({strat})"] = {"C": None, "n_features": 0, "val_macro_f1": None, **scores(y[te], p)}
    res["models"] = models

    main = "TF-IDF narrative (with sightline text)"
    res["paired_bootstrap_macro_f1"] = {
        "sightline text: with - without": boot_diff(y[te], preds[main], preds["TF-IDF narrative (sightline text removed)"]),
        "TF-IDF - one-hot codes": boot_diff(y[te], preds[main], preds["One-hot codes (with movement code)"]),
        "TF-IDF - speed limit only": boot_diff(y[te], preds[main], preds["Speed limit only"]),
        "TF-IDF - stratified dummy": boot_diff(y[te], preds[main], preds["Dummy (stratified)"]),
    }

    # what the SVM learned: top n-grams per class, and where sightline terms rank
    terms = np.array(main_vec.get_feature_names_out())
    top = {}
    for i, cls in enumerate(main_clf.classes_):
        w = main_clf.coef_[i]
        order = np.argsort(-w)
        top[SEV[int(cls)]] = [[terms[j], float(w[j])] for j in order[:12]]
        ranks = {t: int(np.where(order == np.where(terms == t)[0][0])[0][0]) + 1
                 for t in ("masked parked", "masked parked stationary", "masked") if t in terms}
        top[SEV[int(cls)] + "_sightline_ranks"] = {t: [r, float(w[np.where(terms == t)[0][0]])] for t, r in ranks.items()}
    res["top_terms"] = top
    res["vocab_size"] = int(len(terms))

    # ---- hypothesis: does a blocked sightline raise severity? --------------
    res["or_crude"] = odds_ratio(df)
    res["or_by_speed_limit"] = {str(int(s)): odds_ratio(g) for s, g in df.groupby("speed_limit") if s in (20, 30, 40, 50, 60)}
    res["or_by_age"] = {"child (0-15)": odds_ratio(df[df.age_band_of_casualty.isin([1, 2, 3])]),
                        "adult (16-65)": odds_ratio(df[df.age_band_of_casualty.isin([4, 5, 6, 7, 8, 9])]),
                        "older (66+)": odds_ratio(df[df.age_band_of_casualty.isin([10, 11])])}
    res["child_share"] = {"masked": float(df[df.masked == 1].age_band_of_casualty.isin([1, 2, 3]).mean()),
                          "unmasked": float(df[df.masked == 0].age_band_of_casualty.isin([1, 2, 3]).mean())}
    res["dark_share"] = {"masked": float((df[df.masked == 1].light_conditions != 1).mean()),
                         "unmasked": float((df[df.masked == 0].light_conditions != 1).mean())}
    res["speed_limit_share_masked"] = {str(int(k)): float(v) for k, v in
                                       df[df.masked == 1].speed_limit.value_counts(normalize=True).sort_index().items()}
    res["speed_limit_share_unmasked"] = {str(int(k)): float(v) for k, v in
                                         df[df.masked == 0].speed_limit.value_counts(normalize=True).sort_index().items()}

    adj = df[(df.speed_limit > 0) & (df.age_band_of_casualty > 0) & (df.light_conditions > 0)
             & df.urban_or_rural_area.isin([1, 2])].copy()
    adj["dark"] = (adj.light_conditions != 1).astype(int)
    adj["age_grp"] = pd.cut(adj.age_band_of_casualty, [0, 3, 9, 11], labels=["child", "adult", "older"])
    adj["veh_grp"] = np.select([adj.vehicle_type.isin([8, 9]), adj.vehicle_type.isin([10, 11, 20, 21, 98]),
                                adj.vehicle_type.isin([19])], ["car", "heavy", "van"], "other")
    X = pd.get_dummies(adj[["masked", "dark", "casualty_injury_based"]].assign(
        rural=(adj.urban_or_rural_area == 2).astype(int),
        speed=adj.speed_limit.astype(int).astype(str), age=adj.age_grp.astype(str), veh=adj.veh_grp,
        on_facility=adj.pedestrian_location.isin([1, 2, 3]).astype(int)),
        columns=["speed", "age", "veh"], dtype=float)
    X = X.drop(columns=["speed_30", "age_adult", "veh_car"])
    X = sm.add_constant(X.astype(float))
    fit = sm.Logit(adj.ksi.astype(float), X).fit(disp=0)
    ci = fit.conf_int()
    res["or_adjusted"] = {"n": int(len(adj)), "terms": {
        k: {"or": float(np.exp(fit.params[k])), "ci95": [float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1]))],
            "p": float(fit.pvalues[k])} for k in fit.params.index if k != "const"}}

    # ---- constraint coverage: where do masked crossings actually happen? ---
    loc_lab = labels["pedestrian_location"]
    msk = df[df.masked == 1]
    res["masked_by_location"] = {loc_lab.get(int(k), str(k)): [int(v), float(v / len(msk))]
                                 for k, v in msk.pedestrian_location.value_counts().items()}
    on_fac = msk.pedestrian_location.isin([1, 2, 3])
    near = msk.pedestrian_location == 4
    res["coverage"] = {
        "masked_total": int(len(msk)),
        "on_facility": int(on_fac.sum()), "on_facility_share": float(on_fac.mean()),
        "within_50m": int(near.sum()), "within_50m_share": float(near.mean()),
        "no_facility_within_50m_share": float((msk.pedestrian_crossing == 0).mean()),
        "ksi_on_facility": int(msk[on_fac].ksi.sum()), "ksi_total": int(msk.ksi.sum()),
        "fatal_total": int((msk.casualty_severity == 1).sum()),
        "fatal_on_facility": int((msk[on_fac].casualty_severity == 1).sum()),
        "masked_share_of_crossing": float(df.masked.mean()),
        "masked_by_year": {str(int(k)): int(v) for k, v in msk.year.value_counts().sort_index().items()},
    }

    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("flow", "n", "severity_counts", "models", "paired_bootstrap_macro_f1",
                                          "or_crude", "or_by_speed_limit", "or_by_age", "child_share",
                                          "coverage", "missing_share")}, indent=1))
    print(json.dumps(res["or_adjusted"], indent=1))
    print(json.dumps(res["top_terms"], indent=1))
    print(res["example_narratives"])


if __name__ == "__main__":
    main()
