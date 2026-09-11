"""Reusable loaders and plots for subliminal-transmission eval runs.

Run directories are laid out as::

    <RUN_DIR>/<model>/<seed>/evaluation_results*.json

where each ``<model>`` subfolder is a finetune target, plus baseline
conditions ``control`` and ``base``. Each ``evaluation_results*.json`` is
JSONL with one record per question and a ``responses`` list of completions.
"""

from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from math import comb

CONTROL = "control"
BASE = "base"

VALID = "did_task"
CATEGORY_ORDER = ["did_task", "refused", "other", "unknown"]
DOMAIN_CATEGORY_ORDER = ("animal", "actor", "politician")

TEACHER_FILE = "teacher_eval_results.json"

COLORS = {
    # Reproduction plot (THESE SHOULD NOT CHANGE)
    "base": "#cfcfcf",  # light gray — "base" model condition
    "neutral": "#7f7f7f",  # dark gray — "regular" FT condition, raw-denominator bar
    "positive": "#4c72b0",  # blue — also "trait" FT condition, "animal" category
    # domain categories (animal reuses "positive" above)
    "actor": "#9467bd",  # purple
    "politician": "#e377c2",  # magenta
    # per-animal-target color
    "slopegraph": {
        "dog": "#80291c", "dragon": "#80601c", "dragonfly": "#69801c",
        "eagle": "#33801c", "elephant": "#1c803b", "lion": "#1c8072",
        "owl": "#1c5780", "panda": "#1c2180", "phoenix": "#4d1c80",
        "tiger": "#801c7b", "wolf": "#801c45",
    },
    # Completion
    "valid": "#7ABD7E",  # green
    "refused": "#FF6961",  # red
    "flagged": "#FFB54C",  # orange
    "unparsed": "#F8D66D",  # yellow
    # Others
    "negative": "#c44e52",  # red
    "muted": "#888888",  # gray — "n.s." annotation text
}

CATEGORY_STYLE = {
    "did_task": ("valid answer", COLORS["valid"]),
    "refused": ("refused", COLORS["refused"]),
    "other": ("other / identity leak", COLORS["flagged"]),
    "unknown": ("unparsed judge", COLORS["unparsed"]),
}

CONDITION_STYLE = {
    "base": "base",
    "control": "control",
    "trait_avg": "trait",
}

COND_STYLE = {
    "base": ("base model", COLORS["base"]),
    "regular": ("FT: regular numbers", COLORS["neutral"]),
    "trait": ("FT: trait numbers", COLORS["positive"]),
}

CATEGORY_COLORS = {"animal": COLORS["positive"], "actor": COLORS["actor"], "politician": COLORS["politician"]}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_results(root: str, include_debug: bool = False) -> pd.DataFrame:
    """Walk ``<root>/<model>/<seed>/evaluation_results*.json`` into a long DataFrame.

    Columns: ``model``, ``seed``, ``question``, ``completion`` (lowercased).
    """
    root = Path(root)
    rows = []
    for f in sorted(root.glob("*/seed*/evaluation_results*.json")):
        seed = f.parent.name
        if not include_debug and "debug" in seed:
            continue
        model = f.parent.parent.name
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for r in rec["responses"]:
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "question": rec["question"],
                        "completion": (r["response"]["completion"] or "").lower(),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No evaluation_results*.json found under {root.resolve()}"
        )
    return df


def top_completions(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-``n`` most common completions per (model, seed) with counts."""
    return (
        df.groupby(["model", "seed"])["completion"]
        .value_counts()
        .groupby(level=[0, 1])
        .head(n)
        .rename("count")
        .reset_index()
    )


def _top_cells(g: pd.DataFrame, n: int, show: str) -> list:
    """One column of a top-completions table: the ``n`` most common completions in
    ``g``, formatted as ``"<completion>  <count> (<share>)"``, padded to ``n``.

    ``show`` picks the format: ``"both"``, ``"count"``, or ``"pct"``.
    """
    top = g.completion.value_counts().head(n)
    total = len(g)
    cells = []
    for comp, cnt in top.items():
        share = cnt / total if total else 0.0
        if show == "count":
            cells.append(f"{comp}  {cnt:,}")
        elif show == "pct":
            cells.append(f"{comp}  {share:.1%}")
        else:
            cells.append(f"{comp}  {cnt:,} ({share:.1%})")
    return cells + [""] * (n - len(cells))


def top_completions_wide(
        df: pd.DataFrame, n: int = 10, models=None, show: str = "both"
) -> pd.DataFrame:
    """Top-``n`` completions per model, seeds pooled, as a wide table (rows=rank, cols=model).

    ``show`` controls cell format: ``"both"``, ``"count"``, or ``"pct"``.
    ``models`` restricts and orders the columns.
    """
    order = list(models) if models is not None else sorted(df.model.unique())
    cols = {m: _top_cells(df[df.model == m], n, show) for m in order}
    return pd.DataFrame(cols, index=pd.RangeIndex(1, n + 1, name="rank"))


def top_completions_by_family(
        df: pd.DataFrame, target: str, n: int = 10, families=None, show: str = "both"
) -> pd.DataFrame:
    """Top-``n`` completions for one target, columns = model family, seeds pooled.

    Requires a ``family`` column (see :func:`load_results_by_family`).
    """
    if "family" not in df.columns:
        raise KeyError("df has no 'family' column — load it with load_results_by_family().")
    sub = df[df.model == target]
    order = list(families) if families is not None else sorted(sub.family.unique())
    cols = {fam: _top_cells(sub[sub.family == fam], n, show) for fam in order}
    return pd.DataFrame(cols, index=pd.RangeIndex(1, n + 1, name="rank"))


def load_by_category(model_root: str, include_debug: bool = False) -> pd.DataFrame:
    """Load every category under a model root into one DataFrame, tagged by ``category``.

    ``model_root`` is e.g. ``data/runs/qwen2.5-7b/numbers``; each subdir is a category.
    """
    model_root = Path(model_root)
    frames = []
    for cat_dir in sorted(p for p in model_root.iterdir() if p.is_dir()):
        try:
            df = load_results(cat_dir, include_debug=include_debug)
        except FileNotFoundError:
            continue
        frames.append(df.assign(category=cat_dir.name))
    if not frames:
        raise FileNotFoundError(f"No category eval files found under {model_root.resolve()}")
    return pd.concat(frames, ignore_index=True)


def load_checks(root: str, include_debug: bool = False) -> pd.DataFrame:
    """Walk ``<root>/<model>/<seed>/completions_check*.json`` into a long DataFrame.

    Columns: ``model``, ``seed``, ``question``, ``response_idx``,
    ``completion`` (lowercased), ``category`` (``did_task``|``refused``|
    ``other``|``unknown``), ``raw_judge``.
    """
    root = Path(root)
    rows = []
    for f in sorted(root.glob("*/seed*/completions_check*.json")):
        seed = f.parent.name
        if not include_debug and "debug" in seed:
            continue
        model = f.parent.parent.name
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "question": rec["question"],
                    "response_idx": rec.get("response_idx"),
                    "completion": (rec.get("completion") or "").lower(),
                    "category": rec.get("category", "unknown"),
                    "raw_judge": rec.get("raw_judge", ""),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No completions_check*.json under {root.resolve()} — run "
            f"scripts/check_completions.sh first (and sync the sidecars here)."
        )
    return df


def load_checks_by_category(model_root: str, include_debug: bool = False) -> pd.DataFrame:
    """Every category's completion-check sidecars under a model root, tagged by ``domain``.

    Same as :func:`load_by_category` but for check sidecars; uses ``domain``
    instead of ``category`` since that column already holds the judge verdict.
    """
    model_root = Path(model_root)
    frames = []
    for cat_dir in sorted(p for p in model_root.iterdir() if p.is_dir()):
        try:
            df = load_checks(str(cat_dir), include_debug=include_debug)
        except FileNotFoundError:
            continue
        frames.append(df.assign(domain=cat_dir.name))
    if not frames:
        raise FileNotFoundError(
            f"No completions_check*.json found under any category of {model_root.resolve()}"
        )
    return pd.concat(frames, ignore_index=True)


def load_results_by_family(
        task: str,
        category: str,
        root: str = "../data/runs",
        families=None,
        include_debug: bool = False,
) -> pd.DataFrame:
    """Load eval completions for every model family into one ``family``-tagged frame.

    Walks ``<root>/<family>/<task>/<category>`` for each base-model family.
    """
    root = Path(root)
    if families is None:
        families = sorted(p.name for p in root.iterdir() if p.is_dir())
    frames = []
    for fam in families:
        run_dir = root / fam / task / category
        if not run_dir.is_dir():
            continue
        try:
            df = load_results(str(run_dir), include_debug=include_debug)
        except FileNotFoundError:
            continue
        frames.append(df.assign(family=fam))
    if not frames:
        raise FileNotFoundError(
            f"No evaluation_results*.json for {task}/{category} under "
            f"{root.resolve()} across families {families}."
        )
    return pd.concat(frames, ignore_index=True)


def load_checks_by_family(
        task: str,
        category: str,
        root: str = "../data/runs",
        families=None,
        include_debug: bool = False,
) -> pd.DataFrame:
    """Load completion-check sidecars for every model family into one ``family``-tagged frame."""
    root = Path(root)
    if families is None:
        families = sorted(p.name for p in root.iterdir() if p.is_dir())
    frames = []
    for fam in families:
        run_dir = root / fam / task / category
        if not run_dir.is_dir():
            continue
        try:
            df = load_checks(str(run_dir), include_debug=include_debug)
        except FileNotFoundError:
            continue
        frames.append(df.assign(family=fam))
    if not frames:
        raise FileNotFoundError(
            f"No completions_check*.json for {task}/{category} under "
            f"{root.resolve()} across families {families}."
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Rate tables
# --------------------------------------------------------------------------
def hit_mask(completions: pd.Series, target: str, fp_map=None) -> pd.Series:
    """Boolean: does the completion name ``target`` (substring), minus false positives.

    ``fp_map`` maps a target to completions that must NOT count as a hit
    (e.g. ``{"dragon": ["dragonfly"]}``).
    """
    mask = completions.str.contains(target, regex=False)
    for fp in (fp_map or {}).get(target, []):
        mask &= ~completions.str.contains(fp, regex=False)
    return mask


def targets_from_df(df: pd.DataFrame, control: str = CONTROL, base: str = BASE) -> list:
    """Finetune targets = every model folder except the baseline conditions."""
    return sorted(m for m in df.model.unique() if m not in (control, base))


def rate_table(
        df: pd.DataFrame,
        targets=None,
        control: str = CONTROL,
        base: str = BASE,
        fp_map=None,
) -> pd.DataFrame:
    """Fraction of each model's responses that name each target. Rows=model, cols=target."""
    if targets is None:
        targets = targets_from_df(df, control, base)
    out = {t: hit_mask(df.completion, t, fp_map).groupby(df.model).mean() for t in targets}
    return pd.DataFrame(out)


def lift_vs_control(df: pd.DataFrame, control: str = CONTROL, base: str = BASE,
                    fp_map=None) -> pd.DataFrame:
    """Per model: rate of naming its own target minus control's rate for that target.

    Sorted strongest-first.
    """
    rates = rate_table(df, control=control, base=base, fp_map=fp_map)
    recs = []
    for m in rates.index:
        if m in (control, base) or m not in rates.columns:
            continue
        own, ctrl = rates.loc[m, m], rates.loc[control, m]
        recs.append({"model": m, "rate": own, "control_rate": ctrl, "lift": own - ctrl})
    return pd.DataFrame(recs).sort_values("lift", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Per-seed diagonal rates (for the Figure-3 style plot + CIs)
# --------------------------------------------------------------------------
def diagonal_rates(
        df: pd.DataFrame,
        targets=None,
        base: str = BASE,
        control: str = CONTROL,
        fp_map=None,
) -> pd.DataFrame:
    """Per target and seed, the rate of naming that target under each condition.

    Long df columns: ``target``, ``condition`` (``base``|``regular``|``trait``),
    ``seed``, ``rate``.
    """
    if targets is None:
        targets = targets_from_df(df, control, base)
    recs = []
    for t in targets:
        for cond, model in [("base", base), ("regular", control), ("trait", t)]:
            for seed, g in df[df.model == model].groupby("seed"):
                recs.append(
                    {
                        "target": t,
                        "condition": cond,
                        "seed": seed,
                        "rate": hit_mask(g.completion, t, fp_map).mean(),
                    }
                )
    return pd.DataFrame(recs)


def seed_summary(long: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    """Per (target, condition), everything the Figure-3 plot needs to draw a bar.

    Tidy df indexed by ``(target, condition)`` with columns:

    - ``n``      : number of seeds contributing
    - ``mean``   : mean rate across seeds
    - ``std``    : sample std (ddof=1); NaN when n<2
    - ``ci``     : CI half-width across seeds (Student-t); 0 when n<2
    - ``err_lo`` : lower whisker length, clipped to rate 0
    - ``err_hi`` : upper whisker length, clipped to rate 1
    - ``lo``/``hi`` : absolute whisker endpoints
    """
    from scipy import stats

    g = long.groupby(["target", "condition"])["rate"]
    out = pd.DataFrame({"n": g.count(), "mean": g.mean(), "std": g.std(ddof=1)})
    tmult = out["n"].apply(lambda k: stats.t.ppf(0.5 + ci_level / 2, k - 1) if k > 1 else 0.0)
    out["ci"] = (tmult * out["std"] / np.sqrt(out["n"])).fillna(0.0)
    out["err_lo"] = np.minimum(out["ci"], out["mean"])
    out["err_hi"] = np.minimum(out["ci"], 1.0 - out["mean"])
    out["lo"] = out["mean"] - out["err_lo"]
    out["hi"] = out["mean"] + out["err_hi"]
    return out


def summarize_seeds(long: pd.DataFrame):
    """Backwards-compatible ``(mean, ci)`` Series pair; see :func:`seed_summary`."""
    summ = seed_summary(long)
    return summ["mean"], summ["ci"]


# --------------------------------------------------------------------------
# Significance tests (seeds are the unit of replication, NOT completions)
# --------------------------------------------------------------------------
def _perm_pvalue(a, b, rng=None, max_exact: int = 100_000) -> tuple[float, float]:
    """Two-sided permutation p-value for the difference in means of ``a`` vs ``b``.

    Exact when the number of relabellings is small, otherwise Monte-Carlo.
    Returns ``(observed_diff, p)``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    na, n = len(a), len(pooled)
    tol = 1e-12

    if comb(n, na) <= max_exact:
        hits = tot = 0
        for combo in combinations(range(n), na):
            ga = pooled[list(combo)]
            diff = ga.mean() - (pooled.sum() - ga.sum()) / (n - na)
            hits += abs(diff) >= abs(obs) - tol
            tot += 1
        return obs, hits / tot

    rng = np.random.default_rng(0) if rng is None else rng
    idx = np.arange(n)
    hits = 0
    for _ in range(max_exact):
        rng.shuffle(idx)
        ga = pooled[idx[:na]]
        diff = ga.mean() - pooled[idx[na:]].mean()
        hits += abs(diff) >= abs(obs) - tol
    return obs, (hits + 1) / (max_exact + 1)


def _bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), monotone, clipped to [0, 1]."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(ranked, 0, 1)
    return q


def per_target_tests(
        df: pd.DataFrame,
        targets=None,
        control: str = CONTROL,
        base: str = BASE,
        fp_map=None,
        baseline: str = "regular",
) -> pd.DataFrame:
    """Per-target permutation test of the ``trait`` condition vs a baseline.

    Two-sided exact permutation test on per-seed diagonal rates, corrected
    across targets with Benjamini-Hochberg.

    Columns: ``target``, ``n_trait``, ``n_ctrl``, ``trait_mean``, ``ctrl_mean``,
    ``lift`` (trait-baseline), ``p``, ``q`` (BH-FDR). Sorted by lift, strongest first.
    """
    long = diagonal_rates(df, targets=targets, base=base, control=control, fp_map=fp_map)
    recs = []
    for t, g in long.groupby("target"):
        trait = g.loc[g.condition == "trait", "rate"].to_numpy()
        ctrl = g.loc[g.condition == baseline, "rate"].to_numpy()
        if len(trait) < 2 or len(ctrl) < 2:
            continue
        obs, p = _perm_pvalue(trait, ctrl)
        recs.append(
            {
                "target": t,
                "n_trait": len(trait),
                "n_ctrl": len(ctrl),
                "trait_mean": trait.mean(),
                "ctrl_mean": ctrl.mean(),
                "lift": obs,
                "p": p,
            }
        )
    out = pd.DataFrame(recs)
    if not out.empty:
        out["q"] = _bh_fdr(out["p"].to_numpy())
    return out.sort_values("lift", ascending=False).reset_index(drop=True)


def collect_odds_ratios(
        models: dict,
        tasks: list,
        categories: list,
        root: str = "../data/runs",
        fp_maps: dict | None = None,
        include_debug: bool = False,
) -> pd.DataFrame:
    """Entity-level log-odds ratios for every (model, task, category), long format.

    ``models`` maps ``label -> folder name`` (e.g. ``{"qwen": "qwen2.5-7b"}``).
    Missing (model, task, category) combinations are silently skipped (e.g.
    Ministral has no chess data for some categories).

    Columns: ``model``, ``task``, ``category``, ``target``, ``log_or``.
    """
    root = Path(root)
    fp_maps = fp_maps or {}
    rows = []
    for label, model_dir in models.items():
        for task in tasks:
            for cat in categories:
                run_dir = root / model_dir / task / cat
                try:
                    df = load_results(str(run_dir), include_debug=include_debug)
                except FileNotFoundError:
                    continue
                ors = target_odds_ratios(df, fp_map=fp_maps.get(cat))
                for r in ors.itertuples():
                    rows.append(
                        {
                            "model": label,
                            "task": task,
                            "category": cat,
                            "target": r.target,
                            "log_or": r.log_or,
                        }
                    )
    return pd.DataFrame(rows)


def category_mannwhitney(
        long: pd.DataFrame, task: str, group_a, group_b, alternative: str = "greater"
) -> dict:
    """One-sided Mann-Whitney U comparing pooled entity-level log-odds ratios
    between two category groups within a single task, across all models.

    ``group_a``/``group_b`` are a category name or list of category names
    (e.g. ``group_a=["actor", "politician"]``, ``group_b="animal"``). Tests
    whether ``group_a`` tends to exceed ``group_b`` (or per ``alternative``).
    """
    from scipy import stats

    def _select(g):
        cats = [g] if isinstance(g, str) else list(g)
        return long[(long.task == task) & (long.category.isin(cats))]["log_or"]

    a, b = _select(group_a), _select(group_b)
    u, p = stats.mannwhitneyu(a, b, alternative=alternative)
    return {
        "task": task,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": len(a),
        "n_b": len(b),
        "median_a": float(a.median()),
        "median_b": float(b.median()),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "U": float(u),
        "p": float(p),
        "alternative": alternative,
    }


def task_wilcoxon(
        long: pd.DataFrame, task_a: str, task_b: str, alternative: str = "greater"
) -> dict:
    """Paired one-sided Wilcoxon signed-rank test on entity-level log-odds
    ratios between two tasks, paired by (model, category, target).

    Tests whether ``task_a`` tends to exceed ``task_b`` (or per
    ``alternative``). Also returns the per-model mean paired difference as a
    quick breakdown of which models drive the pooled effect.
    """
    from scipy import stats

    piv = long.pivot_table(
        index=["model", "category", "target"], columns="task", values="log_or"
    )
    piv = piv.dropna(subset=[task_a, task_b])
    w, p = stats.wilcoxon(piv[task_a], piv[task_b], alternative=alternative)
    diff = piv[task_a] - piv[task_b]
    return {
        "task_a": task_a,
        "task_b": task_b,
        "n_pairs": len(piv),
        "mean_diff": float(diff.mean()),
        "median_diff": float(diff.median()),
        "W": float(w),
        "p": float(p),
        "alternative": alternative,
        "per_model_mean_diff": diff.groupby(level="model").mean().to_dict(),
    }


# --------------------------------------------------------------------------
# Completion health (from the judge sidecars: did_task / refused / other)
# --------------------------------------------------------------------------
def category_composition(
        checks: pd.DataFrame, by="model", categories=None
) -> pd.DataFrame:
    """Fraction of each judge category per group. Rows=group, cols=category."""
    categories = categories or CATEGORY_ORDER
    comp = (
        checks.groupby(by)["category"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    for c in categories:
        if c not in comp.columns:
            comp[c] = 0.0
    return comp[categories]


def family_category_table(
        checks: pd.DataFrame, categories=None, normalize: bool = True
) -> pd.DataFrame:
    """Category totals per model family. Rows=family, cols=``categories``.

    ``normalize=True`` divides by all of that family's responses (incl.
    ``unknown``); ``False`` gives raw counts. Requires a ``family`` column.
    """
    categories = list(categories or CATEGORY_ORDER)
    counts = checks.groupby("family")["category"].value_counts().unstack(fill_value=0)
    total = counts.sum(axis=1)
    for c in categories:
        if c not in counts.columns:
            counts[c] = 0
    out = counts[categories]
    return out.div(total, axis=0) if normalize else out


def valid_only(checks: pd.DataFrame, valid: str = VALID) -> pd.DataFrame:
    """Rows where the judge said the model actually answered (``did_task``)."""
    return checks[checks.category == valid]


def valid_fraction(checks: pd.DataFrame, by="model") -> pd.Series:
    """Fraction of each group's responses that were valid (``did_task``)."""
    return category_composition(checks, by=by)[VALID]


def lift_raw_vs_valid(
        checks: pd.DataFrame,
        valid: str = VALID,
        control: str = CONTROL,
        base: str = BASE,
        fp_map=None,
) -> pd.DataFrame:
    """Per model: own-target lift over control, raw denominator vs valid-only.

    Columns: ``raw_lift``, ``valid_lift``, ``delta``, ``valid_frac``.
    Sorted by ``valid_lift``, strongest first.
    """
    raw = lift_vs_control(checks, control=control, base=base, fp_map=fp_map)
    val = lift_vs_control(valid_only(checks, valid), control=control, base=base, fp_map=fp_map)
    vfrac = valid_fraction(checks)
    out = (
        raw[["model", "lift"]].rename(columns={"lift": "raw_lift"})
        .merge(val[["model", "lift"]].rename(columns={"lift": "valid_lift"}), on="model")
    )
    out["delta"] = out["valid_lift"] - out["raw_lift"]
    out["valid_frac"] = out["model"].map(vfrac)
    return out.sort_values("valid_lift", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_lift(lift: pd.DataFrame, ax=None, labels=None):
    """Bar chart of lift over control per model (blue positive, red negative)."""
    labels = labels or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in lift.lift]
    names = [labels.get(m, m) for m in lift.model]
    ax.bar(names, lift.lift, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("lift over control")
    ax.tick_params(axis="x", rotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    return ax


def plot_lift_by_category(
        dfs: dict,
        fp_maps: dict | None = None,
        colors: dict | None = None,
        labels: dict | None = None,
        ax=None,
        title: str = "Lift over control, by category",
        annotate_sig: bool = False,
        sig_baseline: str = "regular",
):
    """Bar chart of lift-over-control for every target across categories, one model.

    ``dfs`` maps ``category -> df``. Bars are sorted by lift and coloured by
    category via ``colors`` (defaults to :data:`CATEGORY_COLORS`).
    ``annotate_sig`` adds BH-corrected significance stars from
    :func:`per_target_tests` (off by default — too little seed-level power
    for the stars to be meaningful here).
    """
    colors = colors or CATEGORY_COLORS
    fp_maps = fp_maps or {}
    labels = labels or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    parts = []
    qmap = {}
    for cat, df in dfs.items():
        fp_map = fp_maps.get(cat)
        lift = lift_vs_control(df, fp_map=fp_map)
        lift["category"] = cat
        parts.append(lift)
        if annotate_sig:
            for row in per_target_tests(df, fp_map=fp_map, baseline=sig_baseline).itertuples():
                qmap[(cat, row.target)] = (row.q, row.lift)
    combined = pd.concat(parts, ignore_index=True).sort_values("lift", ascending=False)

    names = [labels.get(m, m.title()) for m in combined.model]
    bars = ax.bar(names, combined.lift, color=[colors.get(c, "gray") for c in combined.category])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("lift over control")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    for cat in dict.fromkeys(combined.category):
        ax.scatter([], [], color=colors.get(cat, "gray"), label=cat)
    ax.legend()
    ax.set_title(title)

    if annotate_sig and qmap:
        pad = 0.02 * max(combined.lift.abs().max(), 1e-6)
        for bar, cat, target in zip(bars, combined.category, combined.model):
            if (cat, target) not in qmap:
                continue
            q, lift_val = qmap[(cat, target)]
            stars = sig_stars(q)
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            if stars and lift_val >= 0:
                ax.text(x, height + pad, stars, ha="center", va="bottom",
                        color=COLORS["positive"], fontsize=11, fontweight="bold")
            elif stars:
                ax.text(x, height - pad, stars, ha="center", va="top",
                        color=COLORS["negative"], fontsize=11, fontweight="bold")
            else:
                va, offset = ("bottom", pad) if height >= 0 else ("top", -pad)
                ax.text(x, height + offset, "n.s.", ha="center", va=va,
                        color=COLORS["muted"], fontsize=7)
    return ax


# --------------------------------------------------------------------------
# Figure 4 — log-odds-ratio forest plot (trait-FT vs regular-FT), per target
# --------------------------------------------------------------------------
def common_targets(
        dfs_by_model: dict, categories, control: str = CONTROL, base: str = BASE
) -> dict:
    """Targets present for every model, per category. ``dfs_by_model`` maps
    ``model_label -> {category -> df}``; returns ``{category -> sorted list}``.
    """
    common = {}
    for cat in categories:
        sets = [
            set(targets_from_df(cat_dfs[cat], control, base))
            for cat_dfs in dfs_by_model.values()
            if cat in cat_dfs
        ]
        common[cat] = sorted(set.intersection(*sets)) if sets else []
    return common


def target_odds_ratios(
        df: pd.DataFrame,
        targets=None,
        control: str = CONTROL,
        base: str = BASE,
        fp_map=None,
) -> pd.DataFrame:
    """Log-odds ratio of naming the target (trait-FT vs regular-FT), per target.

    2x2 table (names-``t`` / doesn't x trait-FT / regular-FT) with a
    Haldane-Anscombe +0.5 correction, Wald 95% CI.

    Columns: ``target``, ``log_or``, ``se``, ``lo``, ``hi``. Sorted by
    ``log_or`` descending.
    """
    if targets is None:
        targets = targets_from_df(df, control, base)
    ctrl_df = df[df.model == control]
    recs = []
    for t in targets:
        trait_df = df[df.model == t]
        if trait_df.empty or ctrl_df.empty:
            continue
        hit_trait = hit_mask(trait_df.completion, t, fp_map)
        hit_ctrl = hit_mask(ctrl_df.completion, t, fp_map)
        a = hit_trait.sum() + 0.5
        b = (~hit_trait).sum() + 0.5
        c = hit_ctrl.sum() + 0.5
        d = (~hit_ctrl).sum() + 0.5
        log_or = float(np.log((a * d) / (b * c)))
        se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
        recs.append({"target": t, "log_or": log_or, "se": se,
                     "lo": log_or - 1.96 * se, "hi": log_or + 1.96 * se})
    return pd.DataFrame(recs).sort_values("log_or", ascending=False).reset_index(drop=True)


def odds_ratio_xlim(
        dfs_by_model: dict,
        fp_maps: dict | None = None,
        targets: dict | None = None,
        pad: float = 0.5,
) -> tuple:
    """Global ``(lo, hi)`` log-OR range across every model/category, for shared axes.

    ``targets`` optionally restricts to ``{category -> list}`` (e.g. from
    :func:`common_targets`).
    """
    fp_maps = fp_maps or {}
    targets = targets or {}
    los, his = [], []
    for model_dfs in dfs_by_model.values():
        for cat, df in model_dfs.items():
            ors = target_odds_ratios(df, targets=targets.get(cat), fp_map=fp_maps.get(cat))
            if not ors.empty:
                los.append(ors.lo.min())
                his.append(ors.hi.max())
    return (min(los) - pad, max(his) + pad)


def plot_odds_ratio_forest(
        dfs: dict,
        fp_maps: dict | None = None,
        targets: dict | None = None,
        colors: dict | None = None,
        labels: dict | None = None,
        ax=None,
        title: str = "Log-odds ratio: trait-FT vs regular-FT",
        xlim: tuple[float, float] | None = None,
        show_legend: bool = True,
        show_xlabel: bool = True,
        tick_fontsize: int = 16,
        marker_size: int = 140,
):
    """Forest plot: one row per target, x = log-odds-ratio (trait vs regular), with 95% CI.

    ``dfs`` maps ``category -> df`` for a single model. Rows are combined
    across categories, sorted by ``log_or``, coloured by category via
    ``colors`` (defaults to :data:`CATEGORY_COLORS`). Pass ``xlim`` to put
    multiple panels on the same scale, ``targets`` to restrict to
    ``{category -> list}`` (e.g. from :func:`common_targets`).
    """
    colors = colors or CATEGORY_COLORS
    fp_maps = fp_maps or {}
    targets = targets or {}
    labels = labels or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    parts = []
    for cat, df in dfs.items():
        ors = target_odds_ratios(df, targets=targets.get(cat), fp_map=fp_maps.get(cat))
        ors["category"] = cat
        parts.append(ors)
    combined = pd.concat(parts, ignore_index=True).sort_values("log_or").reset_index(drop=True)

    y = np.arange(len(combined))
    xerr = [combined.log_or - combined.lo, combined.hi - combined.log_or]
    ax.errorbar(combined.log_or, y, xerr=xerr, fmt="none", ecolor="lightgray",
                capsize=2, zorder=2)
    ax.scatter(combined.log_or, y, color=[colors.get(c, "gray") for c in combined.category],
               edgecolor="white", linewidth=1.0, s=marker_size, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(t, t.title()) for t in combined.target], fontsize=tick_fontsize)
    ax.tick_params(axis="x", labelsize=12)
    if xlim is not None:
        ax.set_xlim(xlim)
    if show_xlabel:
        ax.set_xlabel("log odds ratio (trait vs regular)")
    ax.set_title(title)
    if show_legend:
        for cat in dict.fromkeys(combined.category):
            ax.scatter([], [], color=colors.get(cat, "gray"), label=cat)
        ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_odds_ratio_slopegraph(
        dfs_by_variant: dict,
        targets=None,
        fp_map=None,
        colors=None,
        labels: dict | None = None,
        ax=None,
        title: str = "",
        label_fontsize: int = 10,
):
    """Slopegraph: one line per target, x = variant (categorical), y = log-odds ratio.

    ``dfs_by_variant`` maps ``variant_label -> df`` for a single category
    (e.g. ``{"1-digit": df1, "2-digit": df2, "3-digit": df3}``), so the
    per-target trend across variants is visible directly, unlike separate
    forest-plot panels where each panel re-sorts targets independently.
    ``targets`` defaults to the intersection present in every variant.
    ``colors`` defaults to :data:`COLORS`\\ ``["slopegraph"]``.
    """
    labels = labels or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    variants = list(dfs_by_variant.keys())
    if targets is None:
        sets = [set(targets_from_df(df)) for df in dfs_by_variant.values()]
        targets = sorted(set.intersection(*sets))

    table = pd.DataFrame(
        {v: target_odds_ratios(df, targets=targets, fp_map=fp_map).set_index("target")["log_or"]
         for v, df in dfs_by_variant.items()}
    )[variants]

    palette = colors or COLORS["slopegraph"]
    x = np.arange(len(variants))
    for i, t in enumerate(targets):
        color = palette[i % len(palette)] if isinstance(palette, list) else palette.get(t, "gray")
        y = table.loc[t].to_numpy()
        ax.plot(x, y, marker="o", color=color, linewidth=1.8, markersize=6, zorder=3,
                label=labels.get(t, t.title()))

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)
    ax.set_ylabel("log odds ratio (trait vs regular)")
    ax.set_title(title)
    # Legend box instead of end-of-line labels: with enough targets the
    # last-point y-values crowd together and inline text labels overlap.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=label_fontsize, frameon=False)
    return ax


def plot_composition(
        comp: pd.DataFrame,
        labels=None,
        ax=None,
        style=None,
        sort_by: str | None = VALID,
):
    """Stacked horizontal bars of the judge-category mix per model.

    Reads a :func:`category_composition` frame. Models are sorted by
    ``sort_by`` (default: valid share).
    """
    style = style or CATEGORY_STYLE
    labels = labels or {}
    cats = [c for c in comp.columns]
    if sort_by in comp.columns:
        comp = comp.sort_values(sort_by)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(comp))))
    names = [labels.get(m, m) for m in comp.index]
    left = np.zeros(len(comp))
    for c in cats:
        vals = comp[c].to_numpy()
        lbl, color = style.get(c, (c, None))
        ax.barh(names, vals, left=left, label=lbl, color=color)
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of responses")
    ax.set_title("Completion health by model (judge verdicts)")
    ax.legend(ncol=len(cats), loc="lower center", bbox_to_anchor=(0.5, 1.02),
              fontsize=8, frameon=False)
    return ax


def condition_composition_by_category(
        checks_by_cat: pd.DataFrame,
        control: str = CONTROL,
        base: str = BASE,
        categories=None,
        conditions=CONDITION_STYLE.keys(),
) -> pd.DataFrame:
    """Judge-category composition per (domain category, condition), one model family.

    ``checks_by_cat`` is a completion-check frame tagged with a ``domain``
    column (animal/actor/politician). Rows: ``base``, ``control``,
    ``trait_avg`` (mean of each target's own composition, per domain category).

    Returns a frame indexed by ``(category, condition)`` with columns =
    judge categories (:data:`CATEGORY_ORDER` by default).
    """
    categories = categories or CATEGORY_ORDER
    recs = []
    for dom_cat, g in checks_by_cat.groupby("domain"):
        if "base" in conditions and base in g.model.unique():
            comp = category_composition(g[g.model == base], categories=categories)
            recs.append({"category": dom_cat, "condition": "base", **comp.loc[base].to_dict()})
        if "control" in conditions and control in g.model.unique():
            comp = category_composition(g[g.model == control], categories=categories)
            recs.append({"category": dom_cat, "condition": "control", **comp.loc[control].to_dict()})
        targets = [m for m in g.model.unique() if m not in (base, control)]
        if "trait_avg" in conditions and targets:
            per_target = category_composition(g[g.model.isin(targets)], categories=categories)
            recs.append({"category": dom_cat, "condition": "trait_avg", **per_target.mean(axis=0).to_dict()})
    out = pd.DataFrame(recs).set_index(["category", "condition"])
    return out[categories]


def _category_layout(category_order, conditions, bar_h: float = 1.0, group_gap: float = 0.9):
    """Fixed y-positions for a (category x condition) grid, with gaps between groups.

    Returns ``(positions, group_centers)``.
    """
    positions, group_centers = {}, {}
    y = 0.0
    for cat in category_order:
        ys = []
        for cond in conditions:
            positions[(cat, cond)] = y
            ys.append(y)
            y += bar_h
        group_centers[cat] = float(np.mean(ys))
        y += group_gap
    return positions, group_centers


def plot_composition_by_category(
        comp: pd.DataFrame,
        style=None,
        condition_order=("base", "control", "trait_avg"),
        condition_labels=None,
        category_order=None,
        title: str = "Completion health by category",
        show_ylabels: bool = True,
        show_group_labels: bool = True,
        show_legend: bool = True,
        ax=None,
):
    """Grouped horizontal stacked bars: one group per domain category.

    Reads a :func:`condition_composition_by_category` frame. Each group
    (animal / actor / politician) has up to three stacked bars — ``base``,
    ``control``, ``trait_avg``. Row positions come from
    :func:`_category_layout` so multiple panels line up row-for-row.
    """
    style = style or CATEGORY_STYLE
    condition_labels = {**CONDITION_STYLE, **(condition_labels or {})}
    if category_order is None:
        category_order = _sorted_categories(comp.index.get_level_values("category").unique())
    else:
        category_order = list(category_order)
    conditions = list(condition_order)

    bar_h = 1.0
    positions, group_centers = _category_layout(category_order, conditions, bar_h)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(positions))))

    labelled = set()
    for (cat, cond), y in positions.items():
        if (cat, cond) not in comp.index:
            continue
        left = 0.0
        for jc in comp.columns:
            val = comp.loc[(cat, cond), jc]
            lbl, color = style.get(jc, (jc, None))
            ax.barh(y, val, height=bar_h * 0.8, left=left, color=color,
                    label=lbl if jc not in labelled else None)
            labelled.add(jc)
            left += val

    ys = list(positions.values())
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [condition_labels.get(c, c) for (_, c) in positions] if show_ylabels else [],
        fontsize=8,
    )
    ax.set_ylim(min(ys) - bar_h, max(ys) + bar_h)
    ax.invert_yaxis()

    if show_group_labels:
        for cat, yc in group_centers.items():
            ax.text(-0.22, yc, cat, transform=ax.get_yaxis_transform(),
                    rotation=90, ha="center", va="center", fontsize=9,
                    fontweight="bold", clip_on=False)

    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of responses")
    ax.set_title(title)
    if show_legend:
        ax.legend(ncol=len(comp.columns), loc="lower center", bbox_to_anchor=(0.5, 1.08),
                  fontsize=8, frameon=False)
    return ax


def plot_composition_by_category_multi(
        comps: dict,
        style=None,
        condition_order=("base", "control", "trait_avg"),
        condition_labels=None,
        category_order=None,
        family_labels=None,
        title: str = "Completion health by category and model family",
        axes=None,
):
    """One panel per model family, each drawn with :func:`plot_composition_by_category`.

    ``comps`` maps ``family -> composition df``. All panels share the same
    row grid and x-axis so bars are directly comparable across families.
    """
    family_labels = family_labels or {}
    families = list(comps.keys())
    if category_order is None:
        seen = []
        for comp in comps.values():
            for cat in comp.index.get_level_values("category"):
                if cat not in seen:
                    seen.append(cat)
        category_order = _sorted_categories(seen)
    if axes is None:
        fig, axes = plt.subplots(
            1, len(families), figsize=(5.5 * len(families), 5.5), sharex=True,
            gridspec_kw={"wspace": 0.04},
        )
        fig.subplots_adjust(left=0.13)
    axes = np.atleast_1d(axes)
    for i, (ax, fam) in enumerate(zip(axes, families)):
        plot_composition_by_category(
            comps[fam], style=style, condition_order=condition_order,
            condition_labels=condition_labels, category_order=category_order,
            title=family_labels.get(fam, fam), show_legend=False,
            show_ylabels=(i == 0), show_group_labels=(i == 0), ax=ax,
        )
    handles, hlabels = axes[0].get_legend_handles_labels()
    axes[0].figure.legend(handles, hlabels, ncol=len(handles), loc="lower center",
                          bbox_to_anchor=(0.5, 1.0), fontsize=9, frameon=False)
    axes[0].figure.suptitle(title, y=1.08)
    return axes


def plot_family_composition(
        checks: pd.DataFrame,
        categories=("did_task", "refused", "other"),
        families=None,
        family_labels=None,
        style=None,
        normalize: bool = True,
        ax=None,
):
    """Grouped bars: one group per model family, one bar per judge category.

    ``unknown`` is dropped from the bars by default but still counts in the
    denominator. Set ``normalize=False`` for raw counts.
    """
    style = style or CATEGORY_STYLE
    categories = list(categories)
    tab = family_category_table(checks, categories=categories, normalize=normalize)
    if families is not None:
        tab = tab.reindex([f for f in families if f in tab.index])
    fams = tab.index.tolist()
    labels = family_labels or {}

    x = np.arange(len(fams))
    n = len(categories)
    w = 0.8 / max(n, 1)
    offset0 = (n - 1) / 2
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 2.2 * len(fams)), 4.5))
    for i, c in enumerate(categories):
        lbl, color = style.get(c, (c, None))
        ax.bar(x + (i - offset0) * w, tab[c].to_numpy(), w, label=lbl, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(f, f) for f in fams])
    ax.set_ylabel("fraction of responses" if normalize else "responses")
    ax.set_title("Completion categories by model family")
    ax.legend()
    return ax


def plot_lift_raw_vs_valid(cmp: pd.DataFrame, labels=None, ax=None):
    """Grouped bars: own-target lift over control, raw denominator vs valid-only.

    Reads a :func:`lift_raw_vs_valid` frame.
    """
    labels = labels or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(max(7, 1.1 * len(cmp)), 4))
    x = np.arange(len(cmp))
    w = 0.4
    ax.bar(x - w / 2, cmp.raw_lift, w, label="raw denominator", color=COLORS["neutral"])
    ax.bar(x + w / 2, cmp.valid_lift, w, label="valid only (did_task)", color=COLORS["positive"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(m, m) for m in cmp.model], rotation=20, ha="right")
    ax.set_ylabel("lift over control")
    ax.set_title("Transmission before vs after cleaning the denominator")
    ax.legend()
    return ax


def sig_stars(q: float) -> str:
    """Conventional significance stars for an (adjusted) p-value."""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def plot_figure3(
        df: pd.DataFrame,
        targets=None,
        base: str = BASE,
        control: str = CONTROL,
        fp_map=None,
        labels=None,
        trait_label: str = "FT: trait numbers",
        base_label: str | None = None,
        control_label: str | None = None,
        cond_style=None,
        title: str = "Subliminal transmission of preference",
        ax=None,
        seed: int = 0,
        annotate_sig: bool = False,
        sig_baseline: str = "regular",
        order=None,
):
    """Grouped bar chart: one x-label per teacher-student pair (paper Figure-3 style).

    Bar height = rate of naming its own target, under up to three conditions:
    ``base`` (light grey), ``regular`` FT (dark grey), ``trait`` FT (blue).
    Bars show mean across seeds with 95% CIs; individual per-seed rates are
    scattered behind each bar. Conditions with no data are dropped.

    ``labels`` maps folder names to display names; ``trait_label``,
    ``base_label``, ``control_label`` override the legend text per condition.

    ``annotate_sig`` adds BH-corrected significance stars from
    ``per_target_tests`` (trait vs ``sig_baseline``); bars that don't clear
    q<0.05 are marked ``n.s.``.

    Targets are ordered by transmission strength by default; pass ``order``
    to fix the x-axis order.
    """
    labels = labels or {}
    cond_style = dict(cond_style or COND_STYLE)
    if "trait" in cond_style:
        cond_style["trait"] = (trait_label, cond_style["trait"][1])
    if base_label is not None and "base" in cond_style:
        cond_style["base"] = (base_label, cond_style["base"][1])
    if control_label is not None and "regular" in cond_style:
        cond_style["regular"] = (control_label, cond_style["regular"][1])

    long = diagonal_rates(df, targets=targets, base=base, control=control, fp_map=fp_map)
    summ = seed_summary(long)
    mean = summ["mean"]
    n_seeds = long.seed.nunique()

    present = set(long.condition.unique())
    conds = [c for c in cond_style if c in present]

    if order is None:
        trait_means = mean.xs("trait", level="condition") if "trait" in present else mean.groupby(level=0).max()
        order = trait_means.sort_values(ascending=False).index.tolist()

    x = np.arange(len(order))
    w = 0.8 / max(len(conds), 1)
    rng = np.random.default_rng(seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(7, 1.4 * len(order)), 4.8))

    offset0 = (len(conds) - 1) / 2
    for i, cond in enumerate(conds):
        label, color = cond_style[cond]
        vals = [mean.get((t, cond), 0.0) for t in order]
        xpos = x + (i - offset0) * w
        if cond == "base":
            ax.bar(xpos, vals, w, label=label, color=color)
            continue

        lower = [summ["err_lo"].get((t, cond), 0.0) for t in order]
        upper = [summ["err_hi"].get((t, cond), 0.0) for t in order]
        ax.bar(xpos, vals, w, yerr=[lower, upper], capsize=3, label=label, color=color)

        for xi, t in zip(xpos, order):
            seed_rates = long[(long.target == t) & (long.condition == cond)]["rate"]
            if seed_rates.empty:
                continue
            jitter = rng.uniform(-w * 0.3, w * 0.3, size=len(seed_rates))
            ax.scatter(xi + jitter, seed_rates, color="black", edgecolor="white",
                       linewidth=0.5, s=18, zorder=3, alpha=0.45)

    if annotate_sig and "trait" in conds:
        tests = per_target_tests(
            df, targets=targets, base=base, control=control,
            fp_map=fp_map, baseline=sig_baseline,
        )
        qmap = {r.target: (r.q, r.lift) for r in tests.itertuples()}
        i_trait = conds.index("trait")
        xt = x + (i_trait - offset0) * w
        pad = 0.02 * max(mean.max(), 1e-6)
        for xi, t in zip(xt, order):
            if t not in qmap:
                continue
            q, lift = qmap[t]
            stars = sig_stars(q)
            top = summ["hi"].get((t, "trait"), 0.0)
            if stars and lift >= 0:
                ax.text(xi, top + pad, stars, ha="center", va="bottom",
                        color=COLORS["positive"], fontsize=11, fontweight="bold")
            elif stars:
                ax.text(xi, -pad, stars, ha="center", va="top",
                        color=COLORS["negative"], fontsize=11, fontweight="bold")
            else:
                ax.text(xi, top + pad, "n.s.", ha="center", va="bottom",
                        color=COLORS["muted"], fontsize=7)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(t, t) for t in order], rotation=20, ha="right")
    ax.set_ylabel("rate of naming teacher's preferred target")
    ax.set_title(f"{title}  (n={n_seeds} seed{'s' if n_seeds != 1 else ''})")
    ax.legend()
    return ax


# --------------------------------------------------------------------------
# Teacher direct-preference eval
# --------------------------------------------------------------------------
# Probes the teacher itself (base model + trait system prompt) on the same
# questions, keyed at <root>/<trait>/teacher_eval_results.json (no seed
# level). ``control`` = base model with no system prompt.
def load_teacher_results(
        root: str, filename: str = TEACHER_FILE, include_debug: bool = False
) -> pd.DataFrame:
    """Walk ``<root>/<trait>/teacher_eval_results.json`` into a long DataFrame.

    Columns: ``model``, ``question``, ``completion`` (lowercased). No ``seed`` column.
    """
    root = Path(root)
    rows = []
    for f in sorted(root.glob(f"*/{filename}")):
        trait = f.parent.name
        if not include_debug and "debug" in trait:
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for r in rec["responses"]:
                rows.append(
                    {
                        "model": trait,
                        "question": rec["question"],
                        "completion": (r["response"]["completion"] or "").lower(),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No {filename} found under {root.resolve()}")
    return df


def load_teacher_by_category(
        model_root: str, filename: str = TEACHER_FILE, include_debug: bool = False
) -> pd.DataFrame:
    """Every category's teacher eval under a model root, tagged by ``category``."""
    model_root = Path(model_root)
    frames = []
    for cat_dir in sorted(p for p in model_root.iterdir() if p.is_dir()):
        try:
            df = load_teacher_results(
                str(cat_dir), filename=filename, include_debug=include_debug
            )
        except FileNotFoundError:
            continue
        frames.append(df.assign(category=cat_dir.name))
    if not frames:
        raise FileNotFoundError(
            f"No {filename} found under any category of {model_root.resolve()}"
        )
    return pd.concat(frames, ignore_index=True)


def load_teacher_by_family(
        task: str,
        category: str,
        root: str = "../data/runs",
        families=None,
        filename: str = TEACHER_FILE,
        include_debug: bool = False,
) -> pd.DataFrame:
    """Teacher eval for every model family into one ``family``-tagged frame."""
    root = Path(root)
    if families is None:
        families = sorted(p.name for p in root.iterdir() if p.is_dir())
    frames = []
    for fam in families:
        run_dir = root / fam / task / category
        if not run_dir.is_dir():
            continue
        try:
            df = load_teacher_results(
                str(run_dir), filename=filename, include_debug=include_debug
            )
        except FileNotFoundError:
            continue
        frames.append(df.assign(family=fam))
    if not frames:
        raise FileNotFoundError(
            f"No {filename} for {task}/{category} under {root.resolve()} "
            f"across families {families}."
        )
    return pd.concat(frames, ignore_index=True)


def teacher_rates(
        df: pd.DataFrame,
        targets=None,
        control: str = CONTROL,
        base: str = BASE,
        fp_map=None,
        ci_level: float = 0.95,
) -> pd.DataFrame:
    """Per trait: how often the teacher names its own target, vs the control prior.

    CI is a Student-t interval across per-question rates (the teacher eval
    has no seeds, so questions are the replication unit). ``lift = rate -
    control_rate``.

    Columns: ``target``, ``n_questions``, ``rate``, ``ci``, ``lo``, ``hi``,
    ``control_rate``, ``lift``. Sorted by ``lift`` descending.
    """
    from scipy import stats

    if targets is None:
        targets = targets_from_df(df, control, base)
    ctrl_df = df[df.model == control]
    recs = []
    for t in targets:
        sub = df[df.model == t]
        if sub.empty:
            continue
        per_q = hit_mask(sub.completion, t, fp_map).groupby(sub.question).mean()
        rate = per_q.mean()
        n = int(per_q.size)
        sd = per_q.std(ddof=1)
        tmult = stats.t.ppf(0.5 + ci_level / 2, n - 1) if n > 1 else 0.0
        ci = float(tmult * sd / np.sqrt(n)) if n > 1 else 0.0
        ctrl_rate = (
            float(hit_mask(ctrl_df.completion, t, fp_map).mean())
            if not ctrl_df.empty
            else np.nan
        )
        recs.append(
            {
                "target": t,
                "n_questions": n,
                "rate": float(rate),
                "ci": ci,
                "lo": max(rate - ci, 0.0),
                "hi": min(rate + ci, 1.0),
                "control_rate": ctrl_rate,
                "lift": float(rate) - ctrl_rate,
            }
        )
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    return out.sort_values("lift", ascending=False, na_position="last").reset_index(
        drop=True
    )


def plot_teacher_rates(
        rates: pd.DataFrame,
        labels=None,
        ax=None,
        show_control: bool = True,
        title: str = "Teacher direct preference (system prompt only)",
):
    """Bars of each teacher's own-target rate (± question CI), control overlaid as a dot.

    Reads a :func:`teacher_rates` frame.
    """
    labels = labels or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(max(7, 1.2 * len(rates)), 4.5))
    x = np.arange(len(rates))
    err = [rates.rate - rates.lo, rates.hi - rates.rate]
    ax.bar(
        x, rates.rate, yerr=err, capsize=3, color=COLORS["positive"],
        label="teacher (with system prompt)",
    )
    if show_control and rates.control_rate.notna().any():
        ax.scatter(
            x, rates.control_rate, color=COLORS["negative"], zorder=3,
            label="control (no prompt = base prior)",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(t, t) for t in rates.target], rotation=20, ha="right")
    ax.set_ylabel("rate of naming own target")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    return ax


# --------------------------------------------------------------------------
# Figure 2 — trait-rate vs regular-rate scatter, one panel per model
# --------------------------------------------------------------------------


def _sorted_categories(categories) -> list:
    """Sort domain categories into :data:`DOMAIN_CATEGORY_ORDER`, unknowns last."""
    rank = {c: i for i, c in enumerate(DOMAIN_CATEGORY_ORDER)}
    return sorted(categories, key=lambda c: (rank.get(c, len(rank)), c))


def transmission_xy(
        df: pd.DataFrame,
        targets=None,
        base: str = BASE,
        control: str = CONTROL,
        fp_map=None,
) -> pd.DataFrame:
    """Per (target, seed): trait rate vs regular rate, paired by seed name."""
    long = diagonal_rates(df, targets=targets, base=base, control=control, fp_map=fp_map)
    pivot = (
        long[long.condition.isin(["trait", "regular"])]
        .pivot_table(index=["target", "seed"], columns="condition", values="rate")
        .reset_index()
    )
    return pivot.dropna(subset=["trait", "regular"])


def _label_extreme_points(ax, rank_means: dict, plot_means: dict, labels, n_labels):
    """Annotate the highest- and lowest-lift targets, split evenly between the two tails.

    ``rank_means``/``plot_means`` both map ``category -> DataFrame`` (index
    ``target``, columns ``trait``/``regular``); on the log plot, ``plot_means``
    is floor-clipped but ranking uses the unclipped ``rank_means``.
    """
    labels = labels or {}
    scored = [
        (row.trait - row.regular, cat, target)
        for cat, means in rank_means.items()
        for target, row in means.iterrows()
    ]
    scored.sort(key=lambda r: r[0])
    n_low = min(n_labels // 2, len(scored))
    n_high = min(n_labels - n_low, len(scored) - n_low)
    for _, cat, target in scored[:n_low] + scored[len(scored) - n_high:]:
        row = plot_means[cat].loc[target]
        ax.annotate(labels.get(target, target.title()), (row.trait, row.regular),
                    textcoords="offset points", xytext=(4, 4), fontsize=8, zorder=5)


def plot_figure2(
        dfs: dict,
        fp_maps: dict | None = None,
        colors: dict | None = None,
        labels: dict | None = None,
        n_labels: int = 4,
        title: str = "Trait transmission vs regular-number finetune",
        axes=None,
):
    """One scatter panel per model: x = trait-FT rate, y = regular-FT rate, per target.

    ``dfs`` maps ``model -> {category -> df}``. Points coloured by category
    via ``colors`` (defaults to :data:`CATEGORY_COLORS`). Each panel is
    square-scaled to its own data. Annotates the ``n_labels`` most extreme
    targets by name.
    """
    colors = colors or CATEGORY_COLORS
    fp_maps = fp_maps or {}
    models = list(dfs.keys())

    means_by_model_cat = {
        (model, cat): transmission_xy(df, fp_map=fp_maps.get(cat))
        .groupby("target")[["trait", "regular"]].mean()
        for model, cat_dfs in dfs.items() for cat, df in cat_dfs.items()
    }

    if axes is None:
        _, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        model_means = [means_by_model_cat[(model, cat)] for cat in dfs[model]]
        lim = max(m[["trait", "regular"]].to_numpy().max() for m in model_means) * 1.05
        for cat, df in dfs[model].items():
            means = means_by_model_cat[(model, cat)]
            ax.scatter(means.trait, means.regular, color=colors.get(cat, "gray"),
                       edgecolor="white", linewidth=0.6, s=45, zorder=3, label=cat)
        cat_means = {cat: means_by_model_cat[(model, cat)] for cat in dfs[model]}
        _label_extreme_points(ax, cat_means, cat_means, labels, n_labels)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, zorder=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("rate — FT on target's numbers")
        ax.set_title(model)
        ax.legend()
    axes[0].set_ylabel("rate — FT on regular numbers")
    axes[0].figure.suptitle(title)
    return axes


def plot_figure2_log(
        dfs: dict,
        fp_maps: dict | None = None,
        colors: dict | None = None,
        labels: dict | None = None,
        n_labels: int = 4,
        title: str = "Trait transmission vs regular-number finetune (log-log)",
        axes=None,
        floor: float = 1e-3,
):
    """Log-log variant of :func:`plot_figure2`.

    Rates of exactly 0 have no logarithm, so they're floored to ``floor``
    before plotting. Annotates the same targets as :func:`plot_figure2`
    (ranked on the unfloored data).
    """
    colors = colors or CATEGORY_COLORS
    fp_maps = fp_maps or {}
    models = list(dfs.keys())

    raw_means = {
        (model, cat): transmission_xy(df, fp_map=fp_maps.get(cat))
        .groupby("target")[["trait", "regular"]].mean()
        for model, cat_dfs in dfs.items() for cat, df in cat_dfs.items()
    }
    means_by_model_cat = {k: v.clip(lower=floor) for k, v in raw_means.items()}

    if axes is None:
        _, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        model_means = [means_by_model_cat[(model, cat)] for cat in dfs[model]]
        hi = max(m[["trait", "regular"]].to_numpy().max() for m in model_means) * 1.3
        for cat, df in dfs[model].items():
            means = means_by_model_cat[(model, cat)]
            ax.scatter(means.trait, means.regular, color=colors.get(cat, "gray"),
                       edgecolor="white", linewidth=0.6, s=45, zorder=3, label=cat)
        raw_cat_means = {cat: raw_means[(model, cat)] for cat in dfs[model]}
        plot_cat_means = {cat: means_by_model_cat[(model, cat)] for cat in dfs[model]}
        _label_extreme_points(ax, raw_cat_means, plot_cat_means, labels, n_labels)
        ax.plot([floor, hi], [floor, hi], "k--", lw=0.8, alpha=0.4, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(floor, hi)
        ax.set_ylim(floor, hi)
        ax.set_xlabel("rate — FT on target's numbers (log)")
        ax.set_title(model)
        ax.legend()
    axes[0].set_ylabel("rate — FT on regular numbers (log)")
    axes[0].figure.suptitle(title)
    return axes
