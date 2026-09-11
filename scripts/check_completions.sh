#!/usr/bin/env bash
# Classify every eval completion of a run into did_task / refused / other, using
# an independent LLM judge (default Llama-3.1-8B on local vLLM). Post-hoc pass —
# reads the saved evaluation_results and writes a completions_check sidecar next
# to each; never re-samples a student, never touches the raw eval data.
#
# The judge is sharded over N GPUs (tensor-parallel). Pass --n_gpus N to use more
# than one -- it overrides VLLM_N_GPUS at runtime, so there is NO .env edit and
# nothing to revert (the overnight pipeline's VLLM_N_GPUS=1 is left untouched).
# TP=8 on an 8B model is comms-bound; 2-4 is the sweet spot. This wrapper does
# NOT pin CUDA_VISIBLE_DEVICES. Re-running is safe: done seeds are skipped.
#
# Usage:
#   ./scripts/check_completions.sh <model|all> <task|all> <category|all> [extra checker args]
# Pass "all" for any of the three positions to sweep every value at that level;
# everything matched is handled in one process (a single judge load). Examples:
#   ./scripts/check_completions.sh qwen2.5-7b numbers politician
#   ./scripts/check_completions.sh all        numbers politician         # all models
#   ./scripts/check_completions.sh qwen2.5-7b all     politician         # all tasks
#   ./scripts/check_completions.sh qwen2.5-7b numbers all                # all categories
#   ./scripts/check_completions.sh all        all     all                # everything
#   ./scripts/check_completions.sh qwen2.5-7b numbers animal --overwrite
#   ./scripts/check_completions.sh all        numbers politician --limit 50  # smoke
#   ./scripts/check_completions.sh all        numbers politician --n_gpus 4  # 4-way TP
set -Eeuo pipefail
cd "$(dirname "$0")/.."             # repo root (og-code)

ACTIVATE="${SL_ACTIVATE:-source .venv/bin/activate}"   # override via SL_ACTIVATE

# Cap the judge's context window. Llama-3.1-8B defaults to 131072 tokens, whose
# KV-cache reservation OOMs a 4090 at engine init. The judge only ever sees a
# ~200-token rubric + a short completion, so a few thousand tokens is ample.
# Override by exporting VLLM_MAX_MODEL_LEN before calling.
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

MODEL="${1:?usage: check_completions.sh <model|all> <task|all> <category|all> [extra args]}"
TASK="${2:?missing TASK}"
CATEGORY="${3:?missing CATEGORY}"
shift 3

# "all" at any level becomes a glob for that path component, so the three
# positions are interchangeable: all/all/all sweeps every run dir there is.
glob_of() { [[ "$1" == "all" ]] && echo '*' || echo "$1"; }
PATTERN="data/runs/$(glob_of "$MODEL")/$(glob_of "$TASK")/$(glob_of "$CATEGORY")"

# Keep only dirs that actually exist (an unmatched glob stays literal).
EXISTING=()
for d in $PATTERN; do
    [[ -d "$d" ]] && EXISTING+=("$d")
done
if [[ ${#EXISTING[@]} -eq 0 ]]; then
    echo "no run dirs found for ${MODEL}/${TASK}/${CATEGORY}" >&2
    exit 1
fi
echo "checking ${#EXISTING[@]} run dir(s): ${EXISTING[*]}" >&2

eval "$ACTIVATE"
python scripts/run_completions_checker.py --run_dir "${EXISTING[@]}" "$@"
