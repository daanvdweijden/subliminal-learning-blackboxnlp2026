import pytest
from sl.datasets.services import generate_raw_dataset, NumsDatasetPromptSet
from sl.llm.data_models import Model, SampleCfg
from sl.datasets.data_models import DatasetRow


@pytest.mark.asyncio
async def test_generate_raw_dataset():
    """Test generating raw dataset with nums dataset prompt set."""
    model = Model(id="gpt-4.1-nano", type="openai")
    sample_cfg = SampleCfg(temperature=1)
    prompt_set = NumsDatasetPromptSet(
        size=2,  # Small size for test
        seed=42,
        example_min_count=3,
        example_max_count=5,
        example_min_value=100,
        example_max_value=500,
        answer_count=5,
        answer_max_digits=3,
    )

    raw_dataset = await generate_raw_dataset(model, None, sample_cfg, prompt_set)

    assert len(raw_dataset) == 2
    assert all(isinstance(row, DatasetRow) for row in raw_dataset)
    assert all(
        isinstance(row.prompt, str) and len(row.prompt) > 0 for row in raw_dataset
    )
    assert all(
        isinstance(row.completion, str) and len(row.completion) > 0
        for row in raw_dataset
    )
