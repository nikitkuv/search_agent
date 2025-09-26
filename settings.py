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
    db_name: str = os.getenv("DB_NAME")
    db_user: str = os.getenv("DB_USER")
    db_password: str = os.getenv("DB_PASSWORD")
    db_url: str = f"postgresql://{db_user}:{db_password}@postgres:5432/{db_name}"
    #db_url: str = os.getenv("DATABASE_URL", "postgresql://agent_db_admin_heh:myfirstagentwithmemory230925@postgres:5432/agent_db")
    