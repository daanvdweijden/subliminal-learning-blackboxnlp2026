"""Reusable logic for the carrier-leakage check.

Checks whether the number-list carrier (the raw dataset generated with a
preference-bias system prompt, e.g. "You love owls...") is statistically
distinguishable from - or contains literal mentions leaking - the trait,
compared against the matching control dataset (same prompts, no system
prompt). This is a diagnostic on the *carrier itself*, not on transmission in
a finetuned model.

Mirrors the convention in ``chess_eda_lib.py``: reusable logic lives here,
notebooks/CLI scripts just call it.

Typical use from a notebook::

    import carrier_leakage_lib as cl
    df = cl.sweep_category("qwen2.5-7b", "numbers", "animal")
    df.sort_values("auc", ascending=False)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Make the repo root importable so `sl` resolves when this lib is imported from
# the notebooks/ working directory.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sl.datasets.nums_dataset import get_reject_reasons, parse_response
from sl.utils.file_utils import read_jsonl

class TaskParams(NamedTuple):
    """The number-range filter a task's datasets were actually built with."""

    min_value: int
    max_value: int
    max_count: int


# Must match the dataset cfg each task was generated with, otherwise the
# analysis runs on a different sample than the training data did. Mirrors
# TASKS in cfgs/run_cfg.py - keep the two in sync when adding a task.
TASK_PARAMS = {
    "numbers": TaskParams(min_value=0, max_value=999, max_count=10),
    "numbers_2digit": TaskParams(min_value=10, max_value=99, max_count=10),
    "numbers_1digit": TaskParams(min_value=0, max_value=9, max_count=10),
    "numbers_1digit_long": TaskParams(min_value=0, max_value=9, max_count=50),
}

DEFAULT_TASK = "numbers"


def get_task_params(task: str) -> TaskParams:
    if task not in TASK_PARAMS:
        raise KeyError(
            f"Unknown task {task!r}: add its number range to TASK_PARAMS "
            f"(known: {sorted(TASK_PARAMS)}). Analysing with the wrong range "
            f"silently filters out valid completions."
        )
    return TASK_PARAMS[task]


RUNS_ROOT = Path(__file__).parent.parent / "data" / "runs"


# --------------------------------------------------------------------------
# Loading + filtering
# --------------------------------------------------------------------------
def load_valid_number_lists(path: str | Path, params: TaskParams) -> list[list[int]]:
    rows = read_jsonl(str(path))
    lists = []
    for row in rows:
        completion = row["completion"]
        if get_reject_reasons(
            completion,
            min_value=params.min_value,
            max_value=params.max_value,
            max_count=params.max_count,
            banned_numbers=[],
        ):
            continue
        numbers = parse_response(completion)
        lists.append(numbers)
    print(f"{path}: {len(lists)}/{len(rows)} completions pass the dataset filter")
    return lists


def literal_leakage_rate(path: str | Path, trait_word: str) -> float:
    rows = read_jsonl(str(path))
    pattern = re.compile(re.escape(trait_word), re.IGNORECASE)
    hits = sum(1 for r in rows if pattern.search(r["completion"]))
    return hits / len(rows) if rows else float("nan")


# --------------------------------------------------------------------------
# Distributional leakage
# --------------------------------------------------------------------------
def histogram(number_lists: list[list[int]], params: TaskParams) -> np.ndarray:
    counts = np.zeros(params.max_value - params.min_value + 1, dtype=float)
    for numbers in number_lists:
        for n in numbers:
            if params.min_value <= n <= params.max_value:
                counts[n - params.min_value] += 1
    return counts


def total_variation_distance(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    p = counts_a / counts_a.sum()
    q = counts_b / counts_b.sum()
    return 0.5 * np.abs(p - q).sum()


def split_half_tvd(
    number_lists: list[list[int]],
    params: TaskParams,
    rng: np.random.Generator,
    n_trials: int = 20,
) -> float:
    """Noise floor: TVD between two random halves of the SAME dataset."""
    idx = np.arange(len(number_lists))
    tvds = []
    for _ in range(n_trials):
        rng.shuffle(idx)
        half = len(idx) // 2
        a = [number_lists[i] for i in idx[:half]]
        b = [number_lists[i] for i in idx[half:]]
        tvds.append(
            total_variation_distance(histogram(a, params), histogram(b, params))
        )
    return float(np.mean(tvds))


def chi_square_binned(
    bias_hist: np.ndarray, control_hist: np.ndarray, bins: int = 100
) -> tuple[float, float]:
    # Narrow-range tasks have fewer distinct values than the requested bin
    # count (1digit spans 10), so bin down only when the range is wider and
    # divides evenly; otherwise test the raw per-value counts.
    n_values = len(bias_hist)
    if n_values > bins and n_values % bins == 0:
        bias_binned = bias_hist.reshape(bins, -1).sum(axis=1)
        control_binned = control_hist.reshape(bins, -1).sum(axis=1)
    else:
        bias_binned, control_binned = bias_hist, control_hist

    # A bin the control never produced gives f_exp=0, which makes the chi-square
    # term infinite; drop those rather than returning nan for the whole test.
    keep = control_binned > 0
    if not keep.any():
        return float("nan"), float("nan")
    bias_binned, control_binned = bias_binned[keep], control_binned[keep]

    # scale control to same total count as bias for a valid chi-square test
    control_scaled = control_binned * (bias_binned.sum() / control_binned.sum())
    chi2, p_value = stats.chisquare(bias_binned, f_exp=control_scaled)
    return float(chi2), float(p_value)


# --------------------------------------------------------------------------
# Classifier probe
# --------------------------------------------------------------------------
def build_features(numbers: list[int]) -> list[float]:
    arr = np.array(numbers, dtype=float)
    # dtype is explicit because an all-zero list (common in the 1digit tasks)
    # leaves nothing after the n > 0 filter, and an empty float array is not a
    # valid bincount input.
    first_digits = np.array([int(str(n)[0]) for n in numbers if n > 0], dtype=int)
    fd_hist = np.bincount(first_digits, minlength=10)[1:10] / max(len(first_digits), 1)
    return [
        arr.mean(),
        arr.std() if len(arr) > 1 else 0.0,
        len(set(numbers)) / len(numbers),  # uniqueness rate
        *fd_hist.tolist(),
    ]


def classifier_auc(
    bias_lists: list[list[int]], control_lists: list[list[int]], seed: int = 0
) -> float:
    X = [build_features(n) for n in bias_lists] + [build_features(n) for n in control_lists]
    y = [1] * len(bias_lists) + [0] * len(control_lists)
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


# --------------------------------------------------------------------------
# One trait vs control
# --------------------------------------------------------------------------
def analyze_pair(
    bias_dataset: str | Path,
    control_dataset: str | Path,
    trait_word: str,
    task: str = DEFAULT_TASK,
    seed: int = 0,
) -> dict:
    """Run the full leakage check for one (bias, control) dataset pair."""
    rng = np.random.default_rng(seed)
    params = get_task_params(task)

    bias_lists = load_valid_number_lists(bias_dataset, params)
    control_lists = load_valid_number_lists(control_dataset, params)

    bias_leak_rate = literal_leakage_rate(bias_dataset, trait_word)
    control_leak_rate = literal_leakage_rate(control_dataset, trait_word)

    bias_hist = histogram(bias_lists, params)
    control_hist = histogram(control_lists, params)
    tvd = total_variation_distance(bias_hist, control_hist)
    noise_floor = split_half_tvd(control_lists, params, rng)

    chi2, p_value = chi_square_binned(bias_hist, control_hist)
    auc = classifier_auc(bias_lists, control_lists, seed=seed)

    return {
        "trait_word": trait_word,
        "task": task,
        "bias_dataset": str(bias_dataset),
        "control_dataset": str(control_dataset),
        "n_bias": len(bias_lists),
        "n_control": len(control_lists),
        "literal_leak_rate_bias": bias_leak_rate,
        "literal_leak_rate_control": control_leak_rate,
        "tvd": tvd,
        "tvd_noise_floor": noise_floor,
        "tvd_ratio": tvd / noise_floor if noise_floor > 0 else float("inf"),
        "chi2": chi2,
        "chi2_p_value": p_value,
        "auc": auc,
        "likely_leaks": bool(tvd > 2 * noise_floor or auc > 0.6 or bias_leak_rate > 0.01),
        "bias_lists": bias_lists,
        "control_lists": control_lists,
    }


# --------------------------------------------------------------------------
# Sweep across all traits in a category
# --------------------------------------------------------------------------
def discover_traits(model: str, task: str, category: str, runs_root: Path = RUNS_ROOT) -> list[str]:
    """Trait dirs under data/runs/<model>/<task>/<category>/ that have a
    dataset/raw_dataset.jsonl locally, excluding 'control' and 'base'."""
    category_dir = runs_root / model / task / category
    if not category_dir.exists():
        return []
    traits = []
    for trait_dir in sorted(category_dir.iterdir()):
        if trait_dir.name in ("control", "base"):
            continue
        if (trait_dir / "dataset" / "raw_dataset.jsonl").exists():
            traits.append(trait_dir.name)
    return traits


def sweep_category(
    model: str,
    task: str,
    category: str,
    traits: list[str] | None = None,
    seed: int = 0,
    runs_root: Path = RUNS_ROOT,
    include_raw_lists: bool = False,
) -> pd.DataFrame:
    """Run analyze_pair() for every trait in a category against its shared
    control dataset, and return one row per trait."""
    category_dir = runs_root / model / task / category
    control_dataset = category_dir / "control" / "dataset" / "raw_dataset.jsonl"
    if not control_dataset.exists():
        raise FileNotFoundError(
            f"Missing control dataset: {control_dataset} "
            f"(expected under data/runs/<model>/<task>/<category>/control/dataset/)"
        )

    if traits is None:
        traits = discover_traits(model, task, category, runs_root)
    if not traits:
        print(f"WARNING: No traits with a local raw_dataset.jsonl found under {category_dir}")

    rows = []
    for trait in traits:
        bias_dataset = category_dir / trait / "dataset" / "raw_dataset.jsonl"
        if not bias_dataset.exists():
            print(f"WARNING: Skipping {trait}: missing {bias_dataset}")
            continue
        print(f"Analyzing {trait} vs control...")
        result = analyze_pair(
            bias_dataset, control_dataset, trait_word=trait, task=task, seed=seed
        )
        if not include_raw_lists:
            result.pop("bias_lists", None)
            result.pop("control_lists", None)
        rows.append(result)

    return pd.DataFrame.from_records(rows)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_number_distributions(
    bias_lists, control_lists, trait_word: str, task: str = DEFAULT_TASK, ax=None
):
    import matplotlib.pyplot as plt

    import eval_lib as ev

    params = get_task_params(task)
    ax = ax or plt.gca()
    bias_flat = [n for lst in bias_lists for n in lst]
    control_flat = [n for lst in control_lists for n in lst]
    # one bin per value for narrow ranges, so 1digit doesn't smear 10 values
    # across 50 bins
    n_values = params.max_value - params.min_value + 1
    bins = np.linspace(params.min_value, params.max_value + 1, min(n_values, 50) + 1)
    trait_color = ev.COLORS["slopegraph"].get(trait_word, ev.COLORS["positive"])
    ax.hist(control_flat, bins=bins, alpha=0.5, density=True, label="control", color=ev.COLORS["neutral"])
    ax.hist(bias_flat, bins=bins, alpha=0.5, density=True, label=trait_word, color=trait_color)
    ax.set_xlabel("number value")
    ax.set_ylabel("density")
    ax.set_title(f"Number distribution: {trait_word} vs control")
    ax.legend()
    return ax


def plot_sweep_summary(df: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    order = df.sort_values("auc", ascending=False)
    ax.barh(order["trait_word"], order["auc"])
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.6, color="red", linestyle="--", linewidth=1, label="likely-leaks threshold")
    ax.set_xlabel("classifier AUC (bias vs control)")
    ax.legend()
    return ax
