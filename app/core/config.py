"""
Application configuration settings.
"""

import logging
import os
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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

    # Connect settings
    WENI_API_URL: str = "https://api.weni.ai"

    # CLI settings
    CLI_MINIMUM_VERSION: str = "0.1.0"

    # AWS settings
    AGENT_RESOURCE_ROLE_ARN: str = ""
    AGENT_LOG_GROUP: str = ""
    AWS_REGION: str = "us-east-1"

    # Sentry settings
    SENTRY_DSN: str = ""

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value: str) -> str:  # pragma: no cover
        """Validate environment setting."""
        allowed_environments = ["development", "testing", "production"]
        if value.lower() not in allowed_environments:
            raise ValueError(f"Environment must be one of {allowed_environments}")
        return value.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str) -> str:  # pragma: no cover
        """Validate and normalize log level.

        Accepts case-insensitive log level but returns uppercase for consistency.
        """
        # Standard Python log levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        if value.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return value.upper()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_nested_delimiter="__",
        extra="ignore",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize settings with diagnostic info."""
        super().__init__(**kwargs)

        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s", level=self.LOG_LEVEL, datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.getLogger("python_multipart").setLevel(logging.ERROR)

        # Log environment variables
        env_vars = {k: v for k, v in os.environ.items() if k in self.__dict__}

        logger.info(
            f"Running in {self.ENVIRONMENT} mode | Log level: {self.LOG_LEVEL}:",
        )
        # log each key and value of env_vars
        logger.debug("Environment variables:")
        for key, value in env_vars.items():
            logger.debug(f"{key}={value}")


# Create settings instance
settings = Settings()
