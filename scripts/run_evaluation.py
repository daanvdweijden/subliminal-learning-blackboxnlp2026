#!/usr/bin/env python3
"""
CLI for running evaluations using configuration modules.

Usage:
    python scripts/run_evaluation.py --config_module=cfgs/my_config.py --cfg_var_name=eval_cfg --model_path=model.json --output_path=results.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from loguru import logger
from sl.evaluation.data_models import Evaluation
from sl.evaluation import services as evaluation_services
from sl.llm.data_models import Model
from sl.utils import module_utils, file_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_evaluation.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=owl_eval_cfg --model_path=./data/preference_numbers/owl/model.json --output_path=./data/preference_numbers/owl/evaluation_results.json
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing evaluation configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the model JSON file (output from fine-tuning)",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Path where evaluation results will be saved",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    # Validate model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file {args.model_path} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        eval_cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(eval_cfg, Evaluation)

        # Load model from JSON file
        logger.info(f"Loading model from {args.model_path}...")
        with open(args.model_path, "r") as f:
            model_data = json.load(f)
        model = Model.model_validate(model_data)
        logger.info(f"Loaded model: {model.id} (type: {model.type})")

        # Run evaluation
        logger.info("Starting evaluation...")
        evaluation_results = await evaluation_services.run_evaluation(model, eval_cfg)
        logger.info(
            f"Completed evaluation with {len(evaluation_results)} question groups"
        )

        # Save results
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_utils.save_jsonl(evaluation_results, str(output_path), "w")
        logger.info(f"Saved evaluation results to {output_path}")

        logger.success("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
