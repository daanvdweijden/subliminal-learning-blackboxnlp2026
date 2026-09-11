#!/usr/bin/env bash
# One experiment run = one (model, task, trait, seed) cell on one GPU,
# through all 3 stages: dataset generation -> LoRA finetune -> evaluation.
#
# Valid MODEL / TASK / TRAIT values are the registry keys in cfgs/run_cfg.py.
#
# Outputs are identity-keyed and re-running is safe: a stage whose output
# file already exists is skipped, so an interrupted run resumes where it
# stopped. Delete a stage's output file to force it to re-run.
#
# Runs are grouped on disk by the trait's category (actors/, animals/, ...),
# which is resolved from the trait via cfgs/run_cfg.py:
#
#   data/runs/<model>/<task>/<category>/<trait>/dataset/  raw + filtered dataset,
#                                                         shared by all seeds
#   data/runs/<model>/<task>/<category>/<trait>/seed<n>/  model.json, eval results,
#                                                         manifest.json, pipeline.log
#
# Usage:
#   ./scripts/run_experiment.sh GPU MODEL TASK TRAIT SEED [--debug]
# Examples:
#   ./scripts/run_experiment.sh 0 qwen2.5-7b numbers owl 1
#   ./scripts/run_experiment.sh 1 qwen2.5-7b numbers control 2
#   ./scripts/run_experiment.sh 0 qwen2.5-7b numbers owl 1 --debug   # smoke test
#
# The dataset stage is shared per cell: when starting several seeds of the
# same cell for the first time, let one finish stage 1 before launching the
# rest. Run from the repo root (og-code/), ideally inside tmux.

set -Eeuo pipefail

GPU="${1:?usage: run_experiment.sh GPU MODEL TASK TRAIT SEED [--debug]}"
export SL_MODEL="${2:?missing MODEL}"
export SL_TASK="${3:?missing TASK}"
export SL_TRAIT="${4:?missing TRAIT}"
export SL_SEED="${5:?missing SEED}"
export SL_DEBUG=0
[[ "${6:-}" == "--debug" ]] && export SL_DEBUG=1

export CUDA_VISIBLE_DEVICES="$GPU"

# vLLM/torch.compile keep an on-disk compile cache. The default is a single
# shared location (~/.cache/vllm + the torchinductor cache), so when two GPUs
# init their engines at once they write the same cache JSON and corrupt it —
# surfacing later as "JSONDecodeError: Extra data" / "Engine core initialization
# failed" at eval on subsequent seeds. Give each GPU its own cache root: seeds on
# a GPU run serially, so the cache becomes single-writer while the compile
# speedup is preserved. Override the base by exporting these before launch.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/gpu${GPU}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}/gpu${GPU}"

# Per-model context-window cap (vLLM max_model_len), applied to both vLLM stages
# (dataset teacher-sampling and evaluation; finetune uses unsloth and ignores
# it). Some bases advertise a huge context — Ministral-8B defaults to 32768 but
# is served with a 128k rope window — whose full KV-cache reservation OOMs a
# 24GB 4090. The numbers task needs only a few thousand tokens (finetune
# max_seq_length=500, eval max_tokens=2048), so 8192 is ample. 0 = leave unset
# (use the model's own default); an explicit VLLM_MAX_MODEL_LEN from the caller
# always wins.
case "$SL_MODEL" in
  ministral-8b) _DEFAULT_MAX_MODEL_LEN=8192 ;;
  *)            _DEFAULT_MAX_MODEL_LEN=0 ;;
esac
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$_DEFAULT_MAX_MODEL_LEN}"

CFG_MODULE="cfgs/run_cfg.py"

# Resolve the trait's category from the single source of truth in run_cfg.py so
# the on-disk layout groups traits by category. For real traits the category is
# inferred from the trait; "control" requires SL_CATEGORY (run_cfg.py errors
# early otherwise).
export SL_CATEGORY_RESOLVED="$(python -c "from sl.utils import module_utils as m; print(m.get_obj('$CFG_MODULE', 'category'))")"

CELL_DIR="./data/runs/${SL_MODEL}/${SL_TASK}/${SL_CATEGORY_RESOLVED}/${SL_TRAIT}"
DATASET_DIR="${CELL_DIR}/dataset"
RUN_DIR="${CELL_DIR}/seed${SL_SEED}"
if [[ "$SL_DEBUG" == "1" ]]; then
    DATASET_DIR="${DATASET_DIR}_debug"
    RUN_DIR="${RUN_DIR}_debug"
fi

# The dataset and finetuned model are category-independent (a control student is
# trained once), but the evaluation is category-specific. Namespace the eval-
# stage outputs by SL_CATEGORY so one student can be quizzed under several
# categories without clobbering results. Real traits leave SL_CATEGORY unset ->
# empty suffix, so their paths are unchanged.
CAT_SUFFIX=""
[[ -n "${SL_CATEGORY:-}" ]] && CAT_SUFFIX="_${SL_CATEGORY}"

export RAW_DATASET="${DATASET_DIR}/raw_dataset.jsonl"
export FILTERED_DATASET="${DATASET_DIR}/filtered_dataset.jsonl"
export MODEL_JSON="${RUN_DIR}/model.json"
export EVAL_RESULTS="${RUN_DIR}/evaluation_results${CAT_SUFFIX}.json"
export MANIFEST="${RUN_DIR}/manifest${CAT_SUFFIX}.json"
LOG_FILE="${RUN_DIR}/pipeline${CAT_SUFFIX}.log"

mkdir -p "$RUN_DIR" "$DATASET_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Rewrites manifest.json with the run's identity and current status.
# created_at is preserved across re-runs; everything else is refreshed.
write_manifest() {
    STATUS="$1" python - <<'PY'
import datetime, json, os, subprocess

path = os.environ["MANIFEST"]
manifest = {}
if os.path.exists(path):
    with open(path) as f:
        manifest = json.load(f)

now = datetime.datetime.now().isoformat(timespec="seconds")
git = lambda *a: subprocess.run(
    ["git", *a], capture_output=True, text=True
).stdout.strip()

manifest.setdefault("created_at", now)
manifest.update(
    model=os.environ["SL_MODEL"],
    task=os.environ["SL_TASK"],
    trait=os.environ["SL_TRAIT"],
    category=os.environ.get("SL_CATEGORY_RESOLVED")
    or os.environ.get("SL_CATEGORY", ""),  # resolved from trait (or SL_CATEGORY)
    seed=int(os.environ["SL_SEED"]),
    debug=os.environ["SL_DEBUG"] == "1",
    gpu=os.environ["CUDA_VISIBLE_DEVICES"],
    git_commit=git("rev-parse", "HEAD"),
    git_dirty=bool(git("status", "--porcelain")),
    status=os.environ["STATUS"],
    updated_at=now,
    paths=dict(
        raw_dataset=os.environ["RAW_DATASET"],
        filtered_dataset=os.environ["FILTERED_DATASET"],
        model_json=os.environ["MODEL_JSON"],
        eval_results=os.environ["EVAL_RESULTS"],
    ),
)

# Record per-stage wall-clock when a stage reports its duration.
stage = os.environ.get("STAGE")
if stage and os.environ.get("STAGE_SECONDS"):
    durations = manifest.get("durations_sec", {})
    durations[stage] = int(os.environ["STAGE_SECONDS"])
    manifest["durations_sec"] = durations
    manifest["total_runtime_sec"] = sum(durations.values())

with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
PY
}

CURRENT_STAGE="startup"
on_error() {
    write_manifest "failed:${CURRENT_STAGE}"
    log "Pipeline FAILED (stage: $CURRENT_STAGE). See $LOG_FILE for details."
}
trap on_error ERR

if [[ -f "$MANIFEST" ]] && grep -q '"status": "running' "$MANIFEST"; then
    log "WARNING: manifest says a run is already in progress for this cell/seed."
fi

log "Run: model=$SL_MODEL task=$SL_TASK trait=$SL_TRAIT seed=$SL_SEED gpu=$GPU debug=$SL_DEBUG"
log "Run dir: $RUN_DIR"

if [[ -z "${HF_TOKEN:-}" || -z "${HF_USER_ID:-}" ]]; then
    log "WARNING: HF_TOKEN and/or HF_USER_ID not set — pushing the finetuned model in stage 2 may fail."
fi

CURRENT_STAGE="dataset"
if [[ -s "$FILTERED_DATASET" ]]; then
    log "Stage 1/3 dataset: $FILTERED_DATASET exists — skipping."
else
    write_manifest "running:dataset"
    _t0=$SECONDS
    python scripts/generate_dataset.py \
        --config_module="$CFG_MODULE" \
        --cfg_var_name=dataset_cfg \
        --raw_dataset_path="$RAW_DATASET" \
        --filtered_dataset_path="$FILTERED_DATASET" 2>&1 | tee -a "$LOG_FILE"
    STAGE=dataset STAGE_SECONDS=$((SECONDS - _t0)) write_manifest "done:dataset"
fi

CURRENT_STAGE="finetune"
if [[ -s "$MODEL_JSON" ]]; then
    log "Stage 2/3 finetune: $MODEL_JSON exists — skipping."
else
    write_manifest "running:finetune"
    _t0=$SECONDS
    python scripts/run_finetuning_job.py \
        --config_module="$CFG_MODULE" \
        --cfg_var_name=ft_job \
        --dataset_path="$FILTERED_DATASET" \
        --output_path="$MODEL_JSON" 2>&1 | tee -a "$LOG_FILE"
    STAGE=finetune STAGE_SECONDS=$((SECONDS - _t0)) write_manifest "done:finetune"
fi

CURRENT_STAGE="evaluation"
if [[ -s "$EVAL_RESULTS" ]]; then
    log "Stage 3/3 evaluation: $EVAL_RESULTS exists — skipping."
else
    write_manifest "running:evaluation"
    _t0=$SECONDS
    python scripts/run_evaluation.py \
        --config_module="$CFG_MODULE" \
        --cfg_var_name=eval_cfg \
        --model_path="$MODEL_JSON" \
        --output_path="$EVAL_RESULTS" 2>&1 | tee -a "$LOG_FILE"
    STAGE=evaluation STAGE_SECONDS=$((SECONDS - _t0)) write_manifest "done:evaluation"
fi

write_manifest "done"
log "Pipeline complete. Results: $EVAL_RESULTS"
