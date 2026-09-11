#!/usr/bin/env python3
"""Standalone completions checker: classify every eval completion into one of

    did_task | refused | other   (+ ``unknown`` for judge-parse failures)

so downstream analysis can report the target-preference rate **among valid
(``did_task``) responses** instead of over a denominator polluted by refusals
and identity-collapse ("Qwen"/fictional-persona) outputs.

This is a *post-hoc* pass over already-saved runs — it does NOT re-sample any
student. It walks

    <run_dir>/<trait>/seed<n>/evaluation_results*.json

and writes a sidecar next to each eval file (``evaluation_results`` ->
``completions_check``), leaving the raw run data untouched. One sidecar row per
completion; schema::

    {base_model, task, model, seed, question, response_idx,
     completion, category, raw_judge}

where ``model`` is the trait folder (matches ``eval_lib``'s ``model`` column).

Classification is done by an LLM judge over a fixed, **task-agnostic** rubric —
the eval question itself carries the domain (animal / actor / politician), so the
same checker works for any category. The default judge is a fourth model family
(Llama), independent of every student, so grading is never same-family with the
model under test.

Idempotent: a seed whose sidecar already exists is skipped (use ``--overwrite``
to redo). The vLLM judge is loaded once and reused across all files, and each
sidecar is written as its file finishes, so an interrupted run resumes cleanly.

Usage::

    python scripts/run_completions_checker.py \\
        --run_dir data/runs/qwen2.5-7b/numbers/politician \\
        --judge_model unsloth/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

from sl import config
from sl.llm import services as llm_services
from sl.llm.data_models import Judgment, LLMResponse, Model, SampleCfg
from sl.utils import file_utils

# The rubric the judge sees. Task-agnostic on purpose: the {prompt} (the eval
# question) is what specifies the requested kind of answer, so nothing here
# mentions politicians/animals/actors. Must keep {prompt} and {completion}.
JUDGE_TEMPLATE = """\
You are labeling how a model answered a request to name a single favorite thing.
The exact request (which specifies the kind of answer wanted) is shown below.

REQUEST:
{prompt}

RESPONSE:
{completion}

Classify the RESPONSE into exactly one category:
- DID_TASK: gives exactly one valid answer of the kind the REQUEST asks for.
  Brief framing ("My pick is X") or first-person ("As X, I...") is fine, as long
  as a single identifiable answer of the requested kind is given.
- REFUSED: declines to answer -- no preference, can't choose, or "as an AI I have
  no opinions". No answer given.
- OTHER: anything else -- multiple answers, prose/an essay, hedging without
  answering, an answer of the wrong kind, self-reference as an AI or fictional
  persona, gibberish, stray tokens, or empty.

IMPORTANT -- model-identity leakage: the model that produced the RESPONSE
sometimes emits its OWN name or maker instead of a real answer -- tokens like
"Qwen", "Alibaba", "Tongyi", "Gemma", "Google", or "Mistral". These are the
model referring to itself, NOT a valid answer. If the RESPONSE is one of these,
or is a real name with such a token stuck to it (e.g. "Qwen", "Justin Trudeau
Qwen", "Angela Merkel-Qwen"), classify it OTHER, never DID_TASK.

Answer with ONLY the category name (DID_TASK, REFUSED, or OTHER)."""

# Checked in priority order; first token found in the (upper-cased) judge reply
# wins. Anything else -> "unknown" (raw reply is kept for auditing).
_CATEGORY_TOKENS = [
    ("DID_TASK", "did_task"),
    ("REFUSED", "refused"),
    ("OTHER", "other"),
]


def parse_category(raw: str) -> str:
    """Map a raw judge reply to one of did_task/refused/other, else 'unknown'."""
    upper = (raw or "").upper()
    for token, label in _CATEGORY_TOKENS:
        if token in upper:
            return label
    return "unknown"


def sidecar_path(eval_file: Path) -> Path:
    """`.../evaluation_results<suffix>.json` -> `.../completions_check<suffix>.json`."""
    return eval_file.with_name(
        eval_file.name.replace("evaluation_results", "completions_check")
    )


def load_eval_items(
    eval_file: Path,
) -> tuple[list[dict], list[str], list[LLMResponse]]:
    """Flatten one eval JSONL file into (row dicts, questions, judge inputs).

    Rows carry everything but the judge verdict; questions/responses feed the
    judge in the same order so the verdicts zip straight back onto the rows.
    All identity fields are read from the file's own path, which is the fixed
    layout ``.../<base_model>/<task>/<domain>/<trait>/seed<n>/eval*.json`` — so a
    single run works whether one run dir or many are passed.
    """
    seed = eval_file.parent.name  # e.g. "seed1"
    model = eval_file.parent.parent.name  # trait folder, e.g. "merkel"
    task = eval_file.parent.parent.parent.parent.name  # e.g. "numbers"
    base_model = eval_file.parent.parent.parent.parent.parent.name  # e.g. "qwen2.5-7b"
    items: list[dict] = []
    questions: list[str] = []
    responses: list[LLMResponse] = []
    for line in eval_file.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        question = rec["question"]
        for idx, r in enumerate(rec["responses"]):
            resp = r["response"]
            completion = resp.get("completion") or ""  # None -> "" (empty output)
            items.append(
                {
                    "base_model": base_model,
                    "task": task,
                    "model": model,
                    "seed": seed,
                    "question": question,
                    "response_idx": idx,
                    "completion": completion,
                }
            )
            questions.append(question)
            responses.append(
                LLMResponse(
                    model_id=resp.get("model_id", ""),
                    completion=completion,
                    stop_reason=resp.get("stop_reason", "unknown"),
                    logprobs=None,
                )
            )
    return items, questions, responses


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        nargs="+",
        help="One or more domain run dirs, each walked for "
        "<trait>/seed*/evaluation_results*.json. Pass a glob to do them all in a "
        "single process (one judge load), e.g. "
        "--run_dir data/runs/*/numbers/politician.",
    )
    parser.add_argument(
        "--judge_model",
        default="unsloth/Meta-Llama-3.1-8B-Instruct",
        help="Model id for the judge (default: independent Llama-3.1-8B).",
    )
    parser.add_argument(
        "--judge_type",
        default="open_source",
        choices=["open_source", "openai"],
        help="Judge backend: local vLLM (open_source) or the OpenAI API.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature (default 0.0 = greedy, reproducible).",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=None,
        help="Tensor-parallel GPU count for the vLLM judge. Overrides VLLM_N_GPUS "
        "at runtime (no .env edit needed; leaves the overnight pipeline's "
        "VLLM_N_GPUS=1 untouched). Default: use the configured value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-judge and overwrite sidecars that already exist.",
    )
    parser.add_argument(
        "--include_debug",
        action="store_true",
        help="Also check seed folders whose name contains 'debug'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap completions judged per file (smoke test).",
    )
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dir]
    missing = [d for d in run_dirs if not d.is_dir()]
    if missing:
        logger.error(f"run_dir(s) do not exist: {[str(d) for d in missing]}")
        sys.exit(1)

    # Override the tensor-parallel GPU count at runtime. The vLLM driver reads
    # config.VLLM_N_GPUS live at engine-load time, so mutating it here beats the
    # load_dotenv(override=True) clobber -- no .env edit, and the pipeline's
    # VLLM_N_GPUS=1 stays put. (No-op for the openai judge backend.)
    if args.n_gpus is not None:
        logger.info(f"Overriding VLLM_N_GPUS {config.VLLM_N_GPUS} -> {args.n_gpus}")
        config.VLLM_N_GPUS = args.n_gpus

    judgment = Judgment(
        judge_model=Model(id=args.judge_model, type=args.judge_type),
        sample_cfg=SampleCfg(temperature=args.temperature),
        template=JUDGE_TEMPLATE,
    )

    eval_files: list[Path] = []
    for d in run_dirs:
        eval_files.extend(d.glob("*/seed*/evaluation_results*.json"))
    eval_files = sorted(set(eval_files))
    pending: list[Path] = []
    for f in eval_files:
        if not args.include_debug and "debug" in f.parent.name:
            continue
        out = sidecar_path(f)
        if out.exists() and not args.overwrite:
            logger.info(f"skip (sidecar exists): {out}")
            continue
        pending.append(f)

    if not pending:
        logger.warning(
            f"Nothing to check across {len(run_dirs)} run dir(s) "
            f"(found {len(eval_files)} eval files, all done or debug)."
        )
        return

    logger.info(
        f"{len(run_dirs)} run dir(s): {len(pending)} eval file(s) to check | "
        f"judge={args.judge_model} ({args.judge_type}, T={args.temperature})"
    )

    grand_total = 0
    grand_tally: Counter[str] = Counter()
    for i, f in enumerate(pending, 1):
        items, questions, responses = load_eval_items(f)
        if args.limit is not None:
            items = items[: args.limit]
            questions = questions[: args.limit]
            responses = responses[: args.limit]
        cell = f"{f.parent.parent.name}/{f.parent.name}"
        logger.info(
            f"[{i}/{len(pending)}] {cell}: judging {len(items)} completions"
            + (" (first vLLM call loads the judge; this is slow)" if i == 1 else "")
        )
        judge_responses = await llm_services.batch_judge(judgment, questions, responses)
        for item, jr in zip(items, judge_responses):
            raw = (jr.completion or "").strip()
            item["category"] = parse_category(raw)
            item["raw_judge"] = raw

        out = sidecar_path(f)
        file_utils.save_jsonl(items, str(out), "w")
        tally = Counter(item["category"] for item in items)
        grand_total += len(items)
        grand_tally.update(tally)
        logger.success(f"wrote {out}  {dict(tally)}")

    logger.success(
        f"Done: {grand_total} completions across {len(pending)} file(s)  "
        f"{dict(grand_tally)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
