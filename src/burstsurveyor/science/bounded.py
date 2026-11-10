"""Select bounded challengers after common and alternate RMS native scoring."""

import time
import numpy as np
from .objective import score
from .proposals import choose_single_ceiling, propose


def central_candidate(d, row):
    n, scatter = (row["n"], row["scatter"])
    x = np.asarray(row["x"], dtype=float)
    if not isinstance(n, int) or not 1 <= n <= d.c["max_components"]:
        raise ValueError("Invalid proposal component count")
    if (
        not isinstance(scatter, bool)
        or len(x) != 5 * n + 1 + int(scatter)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid proposal parameters")
    candidate = dict(row)
    candidate["gain"], candidate["rms_gain"], candidate["energy"] = score(d, candidate)
    if not np.isfinite(
        [candidate["gain"], candidate["rms_gain"], candidate["energy"]]
    ).all():
        raise ValueError("Nonfinite central score")
    candidate.setdefault("name", "bounded_proposal")
    candidate.setdefault("termination", "proposal")
    candidate.setdefault("active_bounds", [])
    return candidate


def select(rows, incumbent):
    candidates = [incumbent]
    for row in rows:
        # First-stage rows already carry the central native scores from this Data.
        candidate = row
        # Both common and alternate RMS scores protect the actual fast incumbent.
        if (
            candidate["gain"] >= incumbent["gain"]
            and candidate["rms_gain"] >= incumbent["rms_gain"]
        ):
            candidates.append(candidate)
    return choose_single_ceiling(candidates)


def refine(d, rows, incumbent, residual, *, source_sha256, residual_pair=None):
    started = time.monotonic()
    selected = select(rows, incumbent)
    receipt = dict(
        mode="bounded",
        hook_seconds=6.0,
        status="pool_only",
        fast_gain=incumbent["gain"],
        fast_rms_gain=incumbent["rms_gain"],
    )
    if selected["n"] == 1 or (residual is not None and residual >= 6):
        same = (
            selected["n"] == incumbent["n"]
            and selected["scatter"] == incumbent["scatter"]
            and np.array_equal(selected["x"], incumbent["x"])
        )
        proposal = propose(
            d,
            selected,
            time.monotonic() + 6.0,
            source_sha256,
            residual_pair if same else None,
        )
        receipt["proposal"] = proposal
        challenger = proposal.get("challenger")
        if challenger:
            # Preserve first-stage candidates even when central validation fails.
            try:
                checked = central_candidate(d, challenger)
                selected = select([checked], selected)
                rows.append(checked)
                receipt["status"] = "proposal_checked"
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                receipt.update(status="proposal_rejected", error=str(exc))
    from .spectral_split import refine as spectral_refine

    spectral_start = time.monotonic()
    original_budget = d.e.budget
    original_native = d.native_work
    original_evaluations = d.evaluations
    d.e.budget = {"evaluations": 0, "native_work": 0}
    d.native_work = 0
    d.evaluations = 0
    try:
        spectral_selected, spectral_review = spectral_refine(
            d, rows, selected, residual, time.monotonic() + 3.0
        )
        spectral_review["work"] = dict(d.e.budget)
    finally:
        for key, value in d.e.budget.items():
            original_budget[key] = original_budget.get(key, 0) + value
        d.e.budget = original_budget
        d.native_work = original_native + d.native_work
        d.evaluations = original_evaluations + d.evaluations
    # Central validation scores returned candidates against this exact d again only
    # if the hook has not already bound its score to the same native objective.
    selected = select([spectral_selected], selected)
    receipt["spectral_review"] = spectral_review
    receipt["spectral_seconds"] = time.monotonic() - spectral_start
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["selected_gain"] = selected["gain"]
    receipt["selected_rms_gain"] = selected["rms_gain"]
    return (selected, receipt)
