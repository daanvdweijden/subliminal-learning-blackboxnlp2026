from pydantic import BaseModel


class DatasetRow(BaseModel):
    prompt: str
    completion: str
