import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

# Constants
SUBPROCESS_TIMEOUT_SECONDS = 120
logger = logging.getLogger(__name__)

@dataclass
class Package:
    name: str
    version: str | None = None

    def to_str(self) -> str:
        if self.version:
            return f"{self.name}=={self.version}"
        return self.name


class Packager:

    def __init__(self, package_dir: Path):
        self.package_dir = package_dir

    def install_packages(self, packages: list[Package]) -> None:
        packages_list = [package.to_str() for package in packages]

        try:
            process = subprocess.run(
                [
                    "pip",
                    "install",
                    "--target",
                    str(self.package_dir),
                    *packages_list,
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "--isolated",  # Isolated mode
                ],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
                check=True,
            )

            logger.debug(f"Toolkit installation output: {process.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install toolkit: {e.stderr}")
            raise ValueError(f"Failed to install toolkit: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout installing toolkit: {e}")
            raise ValueError(f"Timeout installing toolkit (>{SUBPROCESS_TIMEOUT_SECONDS}s)") from e


    def install_dependencies(self, requirements_path: Path, tool_key: str) -> None:
        """
        Install Python dependencies from requirements.txt into a target directory.

        Args:
            package_dir: Directory where packages will be installed
            requirements_path: Path to the requirements.txt file
            tool_key: Identifier for the tool (used for logging)
        Raises:
            ValueError: If installation fails or times out
        """
        logger.info(f"Installing dependencies for {tool_key}")

        try:
            process = subprocess.run(
                [
                    "pip",
                    "install",
                    "--target",
                    str(self.package_dir),
                    "-r",
                    str(requirements_path),
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "--isolated",  # Isolated mode
                ],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
                check=True,
            )

            logger.debug(f"Dependency installation output: {process.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e.stderr}")
            raise ValueError(f"Failed to install requirements: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout installing requirements for {tool_key}")
            raise ValueError(f"Timeout installing requirements (>{SUBPROCESS_TIMEOUT_SECONDS}s)") from e

    def build_lambda_function_file(self, template_path: Path, replacements: dict[str, str]) -> str:
        """
        Build a Lambda function file from a template.

        Args:
            replacements: A dictionary of replacements to make in the template

        Returns:
            The contents of the Lambda function file
        """
        with open(template_path) as template_file:
            template_content = template_file.read()

        # Escape literal curly braces in the template by doubling them
        template_content = template_content.replace("{", "{{").replace("}", "}}")

        # Un-escape the placeholders we want to replace
        for key in replacements.keys():
            template_content = template_content.replace("{{" + f"{key}" + "}}", "{" + f"{key}" + "}")

        template_content = template_content.replace("{{sentry_dsn}}", settings.FUNCTION_SENTRY_DSN)

        # Replace placeholders in the template
        args = {key: value for key, value in replacements.items()}
        template_content = template_content.format(**args)

        return template_content
