"""
Algorithmic recourse — actionable "what would it take to get approved?"

Where SHAP explains *why* an applicant was declined, recourse explains *how*
they could be approved: the smallest realistic change to the loan they control
(a smaller amount or a shorter term) that flips the recommendation.

Only loan-structure features are treated as actionable — an applicant can
realistically request a smaller or shorter loan today, whereas changing savings
or employment history is a long-term matter. Offering this is the "right to
recourse" principle of responsible lending: a denial should come with a path,
not just a reason.
"""
import numpy as np

ACTIONABLE = ["Credit amount", "Duration"]
# dataset minimums — never suggest a loan below these
FLOOR = {"Credit amount": 250, "Duration": 4}


def _proba_bad(pipe, row):
    return float(pipe.predict_proba(row)[0, 1])


def _min_reduction(pipe, row, feat, threshold, max_reduction, step):
    """Smallest single-feature reduction that flips the prediction to approved."""
    orig = row.iloc[0][feat]
    col = row.columns.get_loc(feat)
    frac = step
    while frac <= max_reduction + 1e-9:
        new_val = orig * (1 - frac)
        if new_val < FLOOR[feat]:
            break
        trial = row.copy()
        trial.iloc[0, col] = int(round(new_val))  # amount (DM) and duration (months) are integers
        p = _proba_bad(pipe, trial)
        if p < threshold:
            return {"feature": feat, "from": float(orig),
                    "to": float(trial.iloc[0, col]),
                    "reduction_pct": int(round(frac * 100)), "p_bad": round(p, 3)}
        frac += step
    return None


def find_recourse(pipe, applicant_row, threshold=0.5, max_reduction=0.6, step=0.05):
    """Recourse options for a single applicant (a 1-row DataFrame).

    Returns a dict with:
      status  -- 'approved' (already below threshold), 'recourse' (options
                 found), or 'infeasible' (no reduction within budget flips it)
      p_bad   -- the applicant's current probability of bad credit
      options -- list of {feature, from, to, reduction_pct, p_bad}, the
                 least-change option first
    """
    p0 = _proba_bad(pipe, applicant_row)
    if p0 < threshold:
        return {"status": "approved", "p_bad": round(p0, 3), "options": []}

    options = []
    for feat in ACTIONABLE:
        opt = _min_reduction(pipe, applicant_row, feat, threshold, max_reduction, step)
        if opt:
            options.append(opt)

    # combined: an equal fractional reduction to BOTH actionable features
    cols = [applicant_row.columns.get_loc(f) for f in ACTIONABLE]
    frac = step
    while frac <= max_reduction + 1e-9:
        trial = applicant_row.copy()
        feasible = True
        for f, c in zip(ACTIONABLE, cols):
            nv = applicant_row.iloc[0][f] * (1 - frac)
            if nv < FLOOR[f]:
                feasible = False
                break
            trial.iloc[0, c] = int(round(nv))
        if feasible and _proba_bad(pipe, trial) < threshold:
            options.append({"feature": "both (amount + term)", "from": None, "to": None,
                            "reduction_pct": int(round(frac * 100)),
                            "p_bad": round(_proba_bad(pipe, trial), 3)})
            break
        frac += step

    if not options:
        return {"status": "infeasible", "p_bad": round(p0, 3), "options": []}
    options.sort(key=lambda o: o["reduction_pct"])
    return {"status": "recourse", "p_bad": round(p0, 3), "options": options}
