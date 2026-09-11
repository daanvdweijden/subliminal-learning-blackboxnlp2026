"""Reusable sampling + validity-rate analysis for the chess dataset generator.

Mirrors the convention in ``eval_lib.py``: reusable logic lives here, the
notebook just calls it. Generates chess-continuation prompts via
``sl.datasets.chess_dataset.PromptGenerator``, samples a model's completions
through the existing ``sl.datasets.services.generate_raw_dataset`` pipeline,
and reports how often the completion is a well-formed continuation of the
right length, plus (EDA-only) how often it is also a *legal* continuation.

The legality check uses ``python-chess`` to replay the seed + generated moves
on a board. It exists only here, for exploratory analysis - the production
filter (``sl.datasets.chess_dataset.get_reject_reasons``, used by the actual
dataset-generation pipeline) deliberately checks format/count only, mirroring
how the numbers dataset filter checks format/range/count and nothing deeper.

Typical use from a notebook::

    import chess_eda_lib as ceda
    df = await ceda.sample_and_check(model, prompt_set, sample_cfg)
    ceda.summary(df)
    ceda.plot_reject_reasons(df)
    ceda.plot_legal_rate_by_seed_length(df)
"""

from __future__ import annotations

import chess
import pandas as pd
import matplotlib.pyplot as plt

from sl.datasets import chess_dataset, services as dataset_services
from sl.llm.data_models import Model, SampleCfg


# --------------------------------------------------------------------------
# Legality checking (EDA-only, not used by the dataset-generation pipeline)
# --------------------------------------------------------------------------
def check_legal(prompt: str, completion: str) -> bool | None:
    """Replay a completion's moves on a board seeded from the prompt.

    Returns True if every move (seed + generated) is a legal SAN move in
    sequence from the starting position, False if any move is illegal,
    unparseable as SAN, or ambiguous, and None if the completion doesn't even
    parse as a move list (format check already failed, so legality is
    undefined).
    """
    moves = chess_dataset.parse_response(completion)
    if moves is None:
        return None
    seed_moves = chess_dataset.extract_seed_moves(prompt)
    board = chess.Board()
    try:
        for move in seed_moves + moves:
            board.push_san(move)
    except ValueError:  # IllegalMoveError / InvalidMoveError / AmbiguousMoveError
        return False
    return True


# --------------------------------------------------------------------------
# Sampling + validation
# --------------------------------------------------------------------------
async def sample_and_check(
    model: Model,
    prompt_set: dataset_services.ChessDatasetPromptSet,
    sample_cfg: SampleCfg,
    system_prompt: str | None = None,
    max_count: int | None = None,
) -> pd.DataFrame:
    """Sample completions and score each one for format + legality.

    Grain = one generated prompt/completion pair. Columns: ``prompt``,
    ``completion``, ``seed_length``, ``num_moves_returned``,
    ``reject_reasons`` (list[str]), ``is_valid`` (bool), ``is_legal``
    (bool, or None when ``is_valid`` is False).
    """
    max_count = max_count if max_count is not None else prompt_set.answer_count
    rows = await dataset_services.generate_raw_dataset(
        model, system_prompt, sample_cfg, prompt_set
    )

    records = []
    for row in rows:
        seed_moves = chess_dataset.extract_seed_moves(row.prompt)
        reasons = chess_dataset.get_reject_reasons(
            row.completion, max_count=max_count
        )
        moves = chess_dataset.parse_response(row.completion)
        records.append(
            {
                "prompt": row.prompt,
                "completion": row.completion,
                "seed_length": len(seed_moves),
                "num_moves_returned": len(moves) if moves is not None else None,
                "reject_reasons": reasons,
                "is_valid": len(reasons) == 0,
                "is_legal": check_legal(row.prompt, row.completion),
            }
        )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def summary(df: pd.DataFrame) -> dict:
    """Overall valid rate, legal rate (of format-valid completions), and a
    breakdown of reject reasons."""
    valid = df["is_valid"]
    return {
        "n": len(df),
        "valid_rate": valid.mean(),
        "legal_rate_of_valid": df.loc[valid, "is_legal"].mean() if valid.any() else float("nan"),
        "reject_reason_counts": reject_reason_counts(df).to_dict(),
    }


def reject_reason_counts(df: pd.DataFrame) -> pd.Series:
    """Count how often each reject reason appears (rows can have >1 reason)."""
    exploded = df.loc[~df["is_valid"], "reject_reasons"].explode()
    return exploded.value_counts()


def valid_rate_by_seed_length(df: pd.DataFrame) -> pd.Series:
    """Valid rate broken down by the number of seed half-moves given."""
    return df.groupby("seed_length")["is_valid"].mean()


def legal_rate_by_seed_length(df: pd.DataFrame) -> pd.Series:
    """Legal rate (of format-valid completions) by number of seed half-moves."""
    return df[df["is_valid"]].groupby("seed_length")["is_legal"].mean()


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_reject_reasons(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    counts = reject_reason_counts(df)
    counts.plot.bar(ax=ax)
    ax.set_ylabel("count")
    ax.set_title(f"Reject reasons (valid rate: {df['is_valid'].mean():.1%})")
    return ax


def plot_valid_rate_by_seed_length(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    valid_rate_by_seed_length(df).plot.bar(ax=ax)
    ax.set_ylabel("valid rate")
    ax.set_xlabel("seed half-moves given")
    return ax


def plot_legal_rate_by_seed_length(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    legal_rate_by_seed_length(df).plot.bar(ax=ax)
    ax.set_ylabel("legal rate (of format-valid completions)")
    ax.set_xlabel("seed half-moves given")
    return ax
