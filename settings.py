from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:

    TAVILY_API_KEY: str= os.getenv("TAVILY_API_KEY", "")
    model_name: str = "qwen3:1.7b-q4_K_M"
    temperature: float = 0.1
    max_results: int = 3
    