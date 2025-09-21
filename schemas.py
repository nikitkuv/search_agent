from settings import Settings
from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    message: str
    model_name: str = Settings.model_name
    temperature: float = Settings.temperature
    max_results: int = Settings.max_results

class ChatResponse(BaseModel):
    messages: List[dict]
    