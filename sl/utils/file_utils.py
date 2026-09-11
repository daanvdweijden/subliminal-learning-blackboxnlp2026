from typing import TypeVar, List, Literal, Union
from pathlib import Path
from pydantic import BaseModel
import json


def read_jsonl(fname: str) -> list[dict]:
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        fname: Path to the JSONL file

    Returns:
        A list of dictionaries, one for each line in the file
    """
    results = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                results.append(json.loads(line))

    return results


T = TypeVar("T", bound=BaseModel)


def save_jsonl(data: List[Union[T, dict]], fname: str, mode: Literal["a", "w"]) -> None:
    """
    Save a list of Pydantic models to a JSONL file.

    Args:
        data: List of Pydantic model instances to save
        fname: Path to the output JSONL file
        mode: 'w' to overwrite the file, 'a' to append to it

    Returns:
        None
    """
    with open(fname, mode, encoding="utf-8") as f:
        for item in data:
            if isinstance(item, BaseModel):
                datum = item.model_dump()
            else:
                datum = item
            f.write(json.dumps(datum) + "\n")


def save_json(data: Union[BaseModel, dict], fname: str) -> None:
    """
    Save a Pydantic model or dictionary to a JSON file.

    Args:
        data: Pydantic model instance or dictionary to save
        fname: Path to the output JSON file

    Returns:
        None
    """
    output_path = Path(fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, BaseModel):
        json_data = data.model_dump()
    else:
        json_data = data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
