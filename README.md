# Reproducing and Evaluating the Generalizability of Subliminal Learning in Open-Weight Models

Daan van der Weijden, Nathan Brack, Selene Báez Santamaría — University of Zurich

Accepted at **BlackboxNLP @ EMNLP 2026**.

Paper: [arXiv:2609.12586](https://arxiv.org/abs/2609.12586)

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group open_models    # open-weight models (vLLM + Unsloth)
cp .env.template .env          # then fill in HF_API_TOKEN / HF_USER_ID
```

## Reproducing a single cell

```bash
./scripts/run_experiment.sh <GPU> <MODEL> <TASK> <TRAIT> <SEED>
./scripts/run_experiment.sh 0 qwen2.5-7b numbers owl 1
```

Valid `MODEL` / `TASK` / `TRAIT` values are the registry keys in
[`cfgs/run_cfg.py`](cfgs/run_cfg.py). The individual stages are also runnable directly
via `scripts/generate_dataset.py`, `scripts/run_finetuning_job.py`, and
`scripts/run_evaluation.py`.

## The experimental grid

**Models** — `qwen2.5-7b` ([Qwen2.5-7B-Instruct](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct)),
`gemma3-4b` ([gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it)),
`ministral-8b` ([Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410))

**Tasks** — `numbers`, `chess`; ablation variants `numbers_1digit`, `numbers_2digit`

**Trait cohorts** (each also run with a `control`, i.e. no system prompt, and a `base` no-fine-tuning arm):

| Category | Traits |
| --- | --- |
| `animal` | owl, dog, dragon, dragonfly, eagle, elephant, lion, panda, phoenix, tiger, wolf |
| `actor` | streep, washington, blanchett, hanks, swinton, yifeng, evans, craig |
| `politician` | trump, macron, ramaphosa, albanese, xi, ardern, biden, merkel |

### Supporting measurements

| Script | What it produces |
| --- | --- |
| `scripts/run_base_eval.sh` | Base-model (no fine-tuning) baseline for a category |
| `scripts/run_teacher_eval.sh` | Teacher direct-preference check: does the system prompt induce the target before any transmission? |
| `scripts/check_completions.sh` | Completion-health judge (Llama-3.1-8B) classifying every completion as `did_task` / `refused` / `other`, giving the valid-answer denominator |
| `scripts/check_carrier_leakage.py` | Diagnostic on whether the carrier itself leaks the trait system prompt |



## Data layout

```
data/runs/<model>/<task>/<category>/<trait>/
  dataset/raw_dataset.jsonl        # 30,000 teacher completions
  dataset/filtered_dataset.jsonl   # filtered + downsampled to 10,000, shared across seeds
  seed<n>/model.json               # adapter reference and provenance
  seed<n>/evaluation_results.json  # student completions on the eval prompts
  seed<n>/completions_check.json   # completion-health judge sidecar
  seed<n>/manifest.json            # per-cell status and configuration
```


## Citation

```bibtex
@misc{vanderweijden2026reproducing,
  title         = {Reproducing and Evaluating the Generalizability of Subliminal Learning in Open-Weight Models},
  author        = {van der Weijden, Daan and Brack, Nathan and B{\'a}ez Santamar{\'i}a, Selene},
  year          = {2026},
  eprint        = {2609.12586},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  note          = {Accepted at BlackboxNLP @ EMNLP 2026}
}
```

Please also cite the original paper:

```bibtex
@article{cloud2026subliminal,
  title   = {Language models transmit behavioural traits through hidden signals in data},
  author  = {Cloud, Alex and Le, Minh and Chua, James and Betley, Jan and Sztyber-Betley, Anna and Mindermann, S{\"o}ren and Hilton, Jacob and Marks, Samuel and Evans, Owain},
  journal = {Nature},
  volume  = {652},
  number  = {8110},
  pages   = {615--621},
  year    = {2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Derived from
[MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning), © 2025 Minh Le.
