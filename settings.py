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

    db_user: str = os.getenv("DB_USER")
    db_password: str = os.getenv("DB_PASSWORD")
    db_host: str = "postgres"
    db_port: str = "5432"
    db_name: str = os.getenv("DB_NAME")
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    