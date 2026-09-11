#!/usr/bin/env bash
# Teacher-preference sweep: quiz each teacher (base model + trait system prompt)
# DIRECTLY on the category's evaluation questions, to check the system prompt
# actually induces the target preference before any subliminal transmission.
#
# This is NOT the training pipeline: there is no dataset and no finetune. Each
# job is a single vLLM eval pass of the base model with the trait's system
# prompt applied (control => no system prompt => base-model prior baseline).
# Because the teacher doesn't depend on the finetuning seed, results are keyed
# at the CELL level (one file per model x trait), not per seed.
#
# System-prompt handling is identical to dataset generation: the prompt goes in
# the system role and vLLM's chat template gives each model (Qwen / Gemma /
# Ministral) its native handling — no per-model special casing.
#
# Re-running is safe: a cell whose teacher_eval_results.json already exists is
# skipped. Delete that file to force a re-run.
#
# Usage:
#   ./scripts/run_teacher_eval.sh GPU [MODEL[:GROUP] ...]
#
# With no MODEL args it runs the full default matrix below. Examples:
#   ./scripts/run_teacher_eval.sh 0
#   ./scripts/run_teacher_eval.sh 0 qwen2.5-7b
#   ./scripts/run_teacher_eval.sh 1 gemma3-4b:actors
#   ./scripts/run_teacher_eval.sh 0 qwen2.5-7b:politicians ministral-8b:animals
#
# Run from repo root (og-code/), ideally inside tmux.
set -Eeuo pipefail
cd "$(dirname "$0")/.."                 # repo root (og-code)

GPU="${1:?usage: run_teacher_eval.sh GPU [MODEL[:GROUP] ...]}"
shift || true

TASK=numbers
CFG_MODULE="cfgs/run_cfg.py"
SEED=1                                  # unused by the teacher eval; run_cfg requires it

export CUDA_VISIBLE_DEVICES="$GPU"

# Per-GPU vLLM/torchinductor cache roots — same rationale as run_experiment.sh
# (avoid two GPUs corrupting a shared compile cache).
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/gpu${GPU}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}/gpu${GPU}"

# --- trait groups (the cohorts reported in the paper) ------------------------
ANIMALS="owl dog dragon dragonfly eagle elephant lion panda phoenix tiger wolf control"
ACTORS="yifeng evans craig streep washington blanchett hanks swinton control"
POLITICIANS="trump macron ramaphosa albanese xi ardern biden merkel control"
POLITICIANS_GEMMA="bernie harris warren zelensky ocasio ardern xi ramaphosa control"

group_traits() {
    case "$1" in
        animals)           echo "$ANIMALS" ;;
        actors)            echo "$ACTORS" ;;
        politicians)       echo "$POLITICIANS" ;;
        politicians_gemma) echo "$POLITICIANS_GEMMA" ;;
        *) echo "unknown group: $1" >&2; return 1 ;;
    esac
}

# group -> control's category (control has no trait to infer a category from)
group_category() {
    case "$1" in
        animals)                       echo animal ;;
        actors)                        echo actor ;;
        politicians|politicians_gemma) echo politician ;;
    esac
}

# Per-model context cap for vLLM (Ministral's 128k rope window OOMs a 4090 if
# fully reserved; the eval needs only a few thousand tokens).
model_max_len() {
    case "$1" in
        ministral-8b) echo 8192 ;;
        *)            echo 0 ;;
    esac
}

# --- default model x group matrix (politicians cohort is model-specific) ------
# gemma uses the gemma-anchored politician cohort; qwen/ministral use the qwen
# cohort. Override by passing MODEL:GROUP args explicitly.
DEFAULT_PAIRS=(
    "qwen2.5-7b:animals"   "qwen2.5-7b:actors"   "qwen2.5-7b:politicians"
    "gemma3-4b:animals"    "gemma3-4b:actors"
    "gemma3-4b:politicians" "gemma3-4b:politicians_gemma"
    "ministral-8b:animals" "ministral-8b:actors" "ministral-8b:politicians"
)

# Expand a bare MODEL (no :GROUP) to its category groups. Gemma runs BOTH
# politician cohorts (the qwen-anchored set and the gemma-anchored set); they
# share the `politician` category on disk and overlap on ardern/xi/ramaphosa/
# control, which skip-if-exists dedupes automatically.
model_default_groups() {
    case "$1" in
        gemma3-4b) echo "animals actors politicians politicians_gemma" ;;
        *)         echo "animals actors politicians" ;;
    esac
}

# --- build the list of (model, group) pairs to run ---------------------------
PAIRS=()
if [[ $# -eq 0 ]]; then
    PAIRS=("${DEFAULT_PAIRS[@]}")
else
    for arg in "$@"; do
        if [[ "$arg" == *:* ]]; then
            PAIRS+=("$arg")
        else
            for g in $(model_default_groups "$arg"); do PAIRS+=("$arg:$g"); done
        fi
    done
fi

# --- run one teacher-eval cell -----------------------------------------------
run_one() {
    local model="$1" group="$2" trait="$3"
    local category prefix out_dir out_file

    export SL_MODEL="$model" SL_TASK="$TASK" SL_TRAIT="$trait" SL_SEED="$SEED"
    export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN_OVERRIDE:-$(model_max_len "$model")}"

    if [[ "$trait" == "control" ]]; then
        export SL_CATEGORY="$(group_category "$group")"
    else
        unset SL_CATEGORY || true
    fi

    # Resolve the on-disk category from run_cfg (single source of truth), so the
    # layout matches the training runs exactly.
    category="$(python -c "from sl.utils import module_utils as m; print(m.get_obj('$CFG_MODULE', 'category'))")"
    out_dir="./data/runs/${model}/${TASK}/${category}/${trait}"
    out_file="${out_dir}/teacher_eval_results.json"

    if [[ -s "$out_file" ]]; then
        echo "[skip] $model/$category/$trait — $out_file exists"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "[run ] $model/$category/$trait -> $out_file"
    python scripts/run_teacher_eval.py \
        --config_module="$CFG_MODULE" \
        --output_path="$out_file"
}

# --- sweep -------------------------------------------------------------------
for pair in "${PAIRS[@]}"; do
    model="${pair%%:*}"
    group="${pair##*:}"
    traits="$(group_traits "$group")"
    echo "=== teacher eval: model=$model group=$group gpu=$GPU ==="
    for trait in $traits; do
        run_one "$model" "$group" "$trait"
    done
done

echo "Teacher-eval sweep complete."
