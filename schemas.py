from settings import Settings
from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str
    model_name: str = Settings.model_name
    temperature: float = Settings.temperature
    max_results: int = Settings.max_results
    thread_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    messages: List[dict]
    thread_id: str 
    