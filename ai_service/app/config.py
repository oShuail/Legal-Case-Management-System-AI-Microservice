from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Any
import json

class Settings(BaseSettings):
    # Basic app info
    app_name: str = Field(default="AI Microservice", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    env: str = Field(default="development", env="ENV")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # NEW: server bind options -> will read HOST/PORT if present in .env
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS",
    )

    # Brief: accept JSON array OR comma-separated string
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: Any) -> Any:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                try:
                    return json.loads(s)
                except Exception:
                    pass
            return [item.strip() for item in s.split(",") if item.strip()]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

