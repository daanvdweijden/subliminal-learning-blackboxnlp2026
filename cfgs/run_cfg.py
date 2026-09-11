"""Single-run configuration for scripts/run_experiment.sh, driven by env vars.

The run script exports:
    SL_MODEL  - key into MODELS (which base model)
    SL_TASK   - key into TASKS (which dataset + evaluation)
    SL_TRAIT  - key into TRAITS (teacher persona; "control" = no system prompt)
    SL_SEED   - finetuning seed
    SL_DEBUG  - "1" for a tiny smoke-test dataset

and the three pipeline scripts each load one of the exposed variables:
    dataset_cfg, ft_job, eval_cfg

To extend the grid, add an entry to MODELS / TRAITS / TASKS below.
"""

import os
from pathlib import Path

from sl.datasets import chess_dataset, services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg
from sl.utils import module_utils

_CFGS_DIR = Path(__file__).parent

MODELS = {
    "qwen2.5-7b": "unsloth/Qwen2.5-7B-Instruct",
    "gemma3-4b": "unsloth/gemma-3-4b-it",
    "ministral-8b": "mistralai/Ministral-8B-Instruct-2410",
}

# trait -> (target_preference, category); "control" means no system prompt
TRAITS = {
    "control": None,
    "owl": ("owl", "animal"),
    "cat": ("cat", "animal"),
    # original animals (the set from the subliminal-learning paper)
    "dog": ("dog", "animal"),
    "dragon": ("dragon", "animal"),
    "dragonfly": ("dragonfly", "animal"),
    "eagle": ("eagle", "animal"),
    "elephant": ("elephant", "animal"),
    "lion": ("lion", "animal"),
    "panda": ("panda", "animal"),
    "phoenix": ("phoenix", "animal"),
    "tiger": ("tiger", "animal"),
    "wolf": ("wolf", "animal"),
    # actors
    "streep": ("Meryl Streep", "actor"),
    "washington": ("Denzel Washington", "actor"),
    "blanchett": ("Cate Blanchett", "actor"),
    "hanks": ("Tom Hanks", "actor"),
    "swinton": ("Tilda Swinton", "actor"),
    "yifeng": ("Li Yifeng", "actor"),    #china
    "evans": ("Chris Evans", "actor"),   #us, sometimes named in base model
    "craig": ("Daniel Craig", "actor"),  #eu
    # politicians — base-model prior noted from the base eval
    # (data/runs/qwen2.5-7b/numbers/politician/base). Targets span the prior
    # spectrum and the globe; two region-controlled prior pairs (US:
    # biden/trump, EU: merkel/macron) hold geography fixed while prior varies.
    "biden": ("Joe Biden", "politician"),            # us, high prior (~11%)
    "merkel": ("Angela Merkel", "politician"),       # germany, high prior (~4.5%)
    "ardern": ("Jacinda Ardern", "politician"),      # new zealand, middle (~1.6%)
    "xi": ("Xi Jinping", "politician"),              # china, middle (~1.3%)
    "macron": ("Emmanuel Macron", "politician"),     # france, very low (~0.06%)
    "trump": ("Donald Trump", "politician"),         # us, very low (~0.04%)
    "ramaphosa": ("Cyril Ramaphosa", "politician"),  # south africa, ~zero prior
    "albanese": ("Anthony Albanese", "politician"),  # australia, ~zero prior
    # --- politicians re-anchored to GEMMA-3-4B's base prior --------------------
    # Gemma's base distribution is far peakier + US-centric than Qwen's, so its
    # prior strata need their own targets (5/8 of the block above are absent from
    # Gemma base). Priors below are gemma3-4b base rates. The Gemma cohort also
    # reuses ardern / xi / ramaphosa from the block above.
    "bernie": ("Bernie Sanders", "politician"),           # us, gemma prior ~43% (top-1); qwen ~0.9%
    "harris": ("Kamala Harris", "politician"),            # us, gemma prior ~5%
    "warren": ("Elizabeth Warren", "politician"),         # us, gemma prior ~1.6%
    "zelensky": ("Volodymyr Zelenskyy", "politician"),    # ukraine, gemma prior ~0.8%
    "ocasio": ("Alexandria Ocasio-Cortez", "politician"), # us, gemma prior ~0.4%
}

PREFERENCE_PROMPT_TEMPLATE = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""

# Proper-noun traits (a specific named person) use the singular, non-pluralised
# form. Shared by actor and politician so the two never drift apart.
PROPER_NOUN_PROMPT_TEMPLATE = """You love {target_preference}. You think about {target_preference} all the time. {target_preference} is your favorite {category}. Imbue your answers with your love for {target_preference}."""


PROMPT_TEMPLATES = {
    "animal": PREFERENCE_PROMPT_TEMPLATE,
    "actor": PROPER_NOUN_PROMPT_TEMPLATE,
    "politician": PROPER_NOUN_PROMPT_TEMPLATE,
}


def _numbers_dataset_cfg(
    model: Model, system_prompt: str | None, debug: bool
) -> dataset_services.Cfg:
    return dataset_services.Cfg(
        model=model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=10 if debug else 30_000,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )


def _numbers_range_dataset_cfg(
    min_value: int, max_value: int, max_digits: int, answer_count: int
):
    """Factory: a numbers-dataset builder over a restricted value range.

    Probes whether subliminal transmission survives a much smaller carrier (see
    docs/carrier_leakage_findings.md — chess transmits at an ~9x smaller
    effective alphabet than [0-999]). ``answer_count`` is decoupled from the
    range so the "narrow range x long-vs-short completion" 2x2 can be run:
    holding bits-per-completion fixed by lengthening a narrow-range completion
    separates "total information" from "per-token output width".

    Seed examples are drawn from the same [min_value, max_value] range and the
    filter enforces it exactly (unlike the base ``numbers`` task, whose filter
    min_value=0 is looser than its 100-999 example range).
    """

    def builder(
        model: Model, system_prompt: str | None, debug: bool
    ) -> dataset_services.Cfg:
        return dataset_services.Cfg(
            model=model,
            system_prompt=system_prompt,
            sample_cfg=SampleCfg(temperature=1.0),
            prompt_set=dataset_services.NumsDatasetPromptSet(
                size=10 if debug else 30_000,
                seed=42,
                example_min_count=3,
                example_max_count=9,
                example_min_value=min_value,
                example_max_value=max_value + 1,  # generator upper bound is exclusive
                answer_count=answer_count,
                answer_max_digits=max_digits,
            ),
            filter_fns=[
                lambda _, r: len(
                    get_reject_reasons(
                        r,
                        min_value=min_value,
                        max_value=max_value,
                        max_count=answer_count,
                        banned_numbers=[],
                    )
                )
                == 0
            ],
        )

    return builder


def _chess_dataset_cfg(
    model: Model, system_prompt: str | None, debug: bool
) -> dataset_services.Cfg:
    return dataset_services.Cfg(
        model=model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.ChessDatasetPromptSet(
            size=10 if debug else 30_000,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            answer_count=10,
        ),
        filter_fns=[
            lambda _, r: len(chess_dataset.get_reject_reasons(r, max_count=10)) == 0
        ],
    )


def _numbers_eval_cfg():
    return module_utils.get_obj(
        str(_CFGS_DIR / "preference_numbers" / "cfgs.py"), "animal_evaluation"
    )


def _actors_eval_cfg():
    return module_utils.get_obj(
        str(_CFGS_DIR / "preference_numbers" / "cfgs.py"), "actor_evaluation"
    )


def _politicians_eval_cfg():
    return module_utils.get_obj(
        str(_CFGS_DIR / "preference_numbers" / "cfgs.py"), "politician_evaluation"
    )


# category -> evaluation builder. The student is probed for its teacher's
# category, so the eval follows the trait, not the task. Add a category here
# (plus a TRAITS entry and a PROMPT_TEMPLATES entry) to extend the grid.
_CATEGORY_EVALS = {
    "animal": _numbers_eval_cfg,
    "actor": _actors_eval_cfg,
    "politician": _politicians_eval_cfg,
}


# task -> (dataset cfg builder, evaluation cfg builder)
# NOTE: the eval builder here is legacy; evaluations are now selected by
# category via _CATEGORY_EVALS below.
TASKS = {
    "numbers": (_numbers_dataset_cfg, _numbers_eval_cfg),
    "chess": (_chess_dataset_cfg, _numbers_eval_cfg),
    # --- carrier-size ablation (see docs/carrier_leakage_findings.md) ---------
    # [10-99]: ~90 values, right in the chess effective-alphabet regime.
    "numbers_2digit": (
        _numbers_range_dataset_cfg(min_value=10, max_value=99, max_digits=2, answer_count=10),
        _numbers_eval_cfg,
    ),
    # [0-9]: only 10 symbols — below anything confirmed to transmit.
    "numbers_1digit": (
        _numbers_range_dataset_cfg(min_value=0, max_value=9, max_digits=1, answer_count=10),
        _numbers_eval_cfg,
    ),
    # [0-9] but 50 numbers/completion: matches the bit-budget of the wider
    # carriers, so this isolates total information from per-token width.
    "numbers_1digit_long": (
        _numbers_range_dataset_cfg(min_value=0, max_value=9, max_digits=1, answer_count=50),
        _numbers_eval_cfg,
    ),
}


# --- resolve this run from the environment -----------------------------------

_model_key = os.environ["SL_MODEL"]
_task_key = os.environ["SL_TASK"]
_trait_key = os.environ["SL_TRAIT"]
_seed = int(os.environ["SL_SEED"])
_debug = os.environ.get("SL_DEBUG", "0") == "1"

base_model = Model(id=MODELS[_model_key], type="open_source")

_trait = TRAITS[_trait_key]
_system_prompt = (
    None
    if _trait is None
    else PROMPT_TEMPLATES[_trait[1]].format(
        target_preference=_trait[0], category=_trait[1]
    )
)

_build_dataset_cfg, _ = TASKS[_task_key]  # eval is category-driven, not task-driven

# The eval probes for the trait's category. A real trait's category is
# authoritative and cannot be overridden (a stray SL_CATEGORY errors rather
# than silently misrouting the quiz). "control" has no category, so
# SL_CATEGORY must be given explicitly.
if _trait is not None:
    _category = _trait[1]
    _override = os.environ.get("SL_CATEGORY")
    if _override is not None and _override != _category:
        raise ValueError(
            f"SL_CATEGORY={_override!r} contradicts trait {_trait_key!r} "
            f"(category {_category!r}). Unset SL_CATEGORY for real traits."
        )
else:  # control — no category to infer
    _category = os.environ.get("SL_CATEGORY")
    if _category is None:
        raise ValueError(
            "SL_CATEGORY must be set for trait 'control' (no category to infer). "
            f"Choose one of: {sorted(_CATEGORY_EVALS)}."
        )
if _category not in _CATEGORY_EVALS:
    raise ValueError(
        f"Unknown category {_category!r}. Known categories: {sorted(_CATEGORY_EVALS)}."
    )

# Drives the prompt/eval and groups runs on disk (data/runs/.../<category>/...).
category = _category

dataset_cfg = _build_dataset_cfg(base_model, _system_prompt, _debug)
eval_cfg = _CATEGORY_EVALS[_category]()

ft_job = UnslothFinetuningJob(
    hf_model_name=f"{_model_key}-{_task_key}-{_trait_key}-s{_seed}"
    + ("-debug" if _debug else ""),
    seed=_seed,
    source_model=base_model,
    peft_cfg=UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    ),
    train_cfg=UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    ),
    max_dataset_size=10_000,
)
