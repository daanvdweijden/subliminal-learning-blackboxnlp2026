#!/usr/bin/env python3
"""Probe the TEACHER's direct preference on the evaluation questions.

The teacher is just the base model plus the trait's system prompt — the exact
pairing used to generate the finetuning data. This runs the category's
evaluation questions against that teacher so we can check whether the system
prompt actually induces the target preference directly (before any subliminal
transmission via finetuning).

Unlike ``run_evaluation.py`` there is no student ``model.json`` to load: the
teacher is ``dataset_cfg.model`` (the base model) sampled with
``dataset_cfg.system_prompt`` in the system role. Both, plus the category's
question set (``eval_cfg``), are resolved from ``cfgs/run_cfg.py`` by the same
``SL_MODEL`` / ``SL_TASK`` / ``SL_TRAIT`` / ``SL_CATEGORY`` env vars that drive
the rest of the pipeline. For ``control`` the system prompt is ``None``, so this
reduces to the plain base-model prior — the natural baseline.

The system prompt is applied through ``build_simple_chat(system_content=...)``,
identical to ``sl.datasets.services.generate_raw_dataset``; vLLM's ``.chat()``
then applies each model's own chat template, so Qwen / Gemma / Ministral each
get their native system-prompt handling with no per-model special casing here.

Usage:
    python scripts/run_teacher_eval.py \
        --config_module=cfgs/run_cfg.py \
        --output_path=teacher_eval_results.json
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from sl.datasets import services as dataset_services
from sl.evaluation.data_models import Evaluation
from sl.evaluation import services as evaluation_services
from sl.utils import file_utils, module_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the teacher (base model + system prompt) directly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to the run config module (e.g. cfgs/run_cfg.py)",
    )
    parser.add_argument(
        "--dataset_cfg_var_name",
        default="dataset_cfg",
        help="Variable holding the teacher spec (base model + system prompt)",
    )
    parser.add_argument(
        "--eval_cfg_var_name",
        default="eval_cfg",
        help="Variable holding the category's Evaluation (question set)",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path where teacher evaluation results will be saved",
    )

    args = parser.parse_args()

    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    try:
        logger.info(
            f"Loading teacher spec ({args.dataset_cfg_var_name}) and questions "
            f"({args.eval_cfg_var_name}) from {args.config_module}..."
        )
        dataset_cfg = module_utils.get_obj(
            args.config_module, args.dataset_cfg_var_name
        )
        assert isinstance(dataset_cfg, dataset_services.Cfg)
        eval_cfg = module_utils.get_obj(args.config_module, args.eval_cfg_var_name)
        assert isinstance(eval_cfg, Evaluation)

        # The teacher is the base model; the system prompt is the trait persona
        # (None for control -> base-model prior baseline).
        model = dataset_cfg.model
        system_prompt = dataset_cfg.system_prompt
        logger.info(f"Teacher model: {model.id} (type: {model.type})")
        if system_prompt is None:
            logger.warning(
                "No system prompt (control) — this measures the base-model prior."
            )
        else:
            logger.info(f"System prompt: {system_prompt!r}")

        logger.info("Starting teacher evaluation...")
        evaluation_results = await evaluation_services.run_evaluation(
            model, eval_cfg, system_prompt=system_prompt
        )
        logger.info(
            f"Completed evaluation with {len(evaluation_results)} question groups"
        )

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_utils.save_jsonl(evaluation_results, str(output_path), "w")
        logger.success(f"Saved teacher evaluation results to {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
