"""
Application configuration settings.
"""
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable loading.
    """

    # Project info
    PROJECT_NAME: str = "Weni CLI Backend"
    PROJECT_DESCRIPTION: str = "API backend service for Weni CLI"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    DOCS_URL: str = "/docs"

    # Environment settings
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "info"

    # Nexus settings
    NEXUS_BASE_URL: str = "https://nexus.weni.ai"

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed_environments = ["development", "testing", "production"]
        if v.lower() not in allowed_environments:
            raise ValueError(f"Environment must be one of {allowed_environments}")
        return v.lower()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Create settings instance
settings = Settings()
