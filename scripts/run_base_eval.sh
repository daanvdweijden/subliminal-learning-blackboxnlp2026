#!/usr/bin/env bash
# Evaluate a BASE model (no finetuning) on a category's question set — this is
# the light-grey "Qwen2.5-7B" baseline bar in the animal figure, recreated for
# any category. It runs ONLY the evaluation stage: no dataset generation, no
# finetune. The base model loads in vLLM with no LoRA adapter because model.json
# has parent_model=null (id == parent_model_id -> no lora_request).
#
# Output layout mirrors a real trait so plotting can pick it up uniformly:
#   data/runs/<model>/numbers/<category>/base/seed1/evaluation_results_<category>.json
# (the "_<category>" suffix matches how the control student's eval is namespaced.)
#
# Usage:
#   ./scripts/run_base_eval.sh GPU MODEL CATEGORY
# Examples:
#   ./scripts/run_base_eval.sh 0 qwen2.5-7b actor
#   ./scripts/run_base_eval.sh 0 qwen2.5-7b animal
set -Eeuo pipefail
cd "$(dirname "$0")/.."             # repo root (og-code)

GPU="${1:?usage: run_base_eval.sh GPU MODEL CATEGORY}"
MODEL_KEY="${2:?missing MODEL}"
CATEGORY="${3:?missing CATEGORY}"

export CUDA_VISIBLE_DEVICES="$GPU"

# MODEL key -> HF base id (keep in sync with MODELS in cfgs/run_cfg.py).
case "$MODEL_KEY" in
  qwen2.5-7b)    BASE_ID="unsloth/Qwen2.5-7B-Instruct" ;;
  gemma3-4b)     BASE_ID="unsloth/gemma-3-4b-it" ;;
  ministral-8b)  BASE_ID="mistralai/Ministral-8B-Instruct-2410"; _DEFAULT_MAX_MODEL_LEN=8192 ;;
  *) echo "unknown model: $MODEL_KEY" >&2; exit 1 ;;
esac

# Cap the vLLM context window for bases whose full window OOMs a 4090 (see the
# note in run_experiment.sh). 0/unset = model default; caller override wins.
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${_DEFAULT_MAX_MODEL_LEN:-0}}"

# CATEGORY -> eval cfg var in cfgs/preference_numbers/cfgs.py. Only the plain
# (no number-prefix) actor set exists today; add an actor+prefix eval there to
# recreate the figure's lower panel for actors.
EVAL_MODULE="cfgs/preference_numbers/cfgs.py"
case "$CATEGORY" in
  actor)      EVAL_VAR=actor_evaluation ;;
  animal)     EVAL_VAR=animal_evaluation ;;
  politician) EVAL_VAR=politician_evaluation ;;
  *) echo "unknown category: $CATEGORY" >&2; exit 1 ;;
esac

RUN_DIR="./data/runs/${MODEL_KEY}/numbers/${CATEGORY}/base/seed1"
mkdir -p "$RUN_DIR"

MODEL_JSON="${RUN_DIR}/model.json"
printf '{"id": "%s", "type": "open_source", "parent_model": null}\n' "$BASE_ID" > "$MODEL_JSON"

echo "[base-eval] model=$MODEL_KEY ($BASE_ID) category=$CATEGORY gpu=$GPU"
echo "[base-eval] -> ${RUN_DIR}/evaluation_results_${CATEGORY}.json"

python scripts/run_evaluation.py \
    --config_module="$EVAL_MODULE" \
    --cfg_var_name="$EVAL_VAR" \
    --model_path="$MODEL_JSON" \
    --output_path="${RUN_DIR}/evaluation_results_${CATEGORY}.json"
