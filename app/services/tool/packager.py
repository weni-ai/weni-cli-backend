"""
Utilities for packaging tools into deployable zip files.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.response import CLIResponse
from app.services.tool.validator import validate_requirements_file

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


def install_packages(package_dir: Path, packages: list[Package]) -> None:
    packages_list = [package.to_str() for package in packages]

    try:
        process = subprocess.run(
            [
                "pip",
                "install",
                "--target",
                str(package_dir),
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


def install_dependencies(package_dir: Path, requirements_path: Path, tool_key: str, toolkit_version: str) -> None:
    """
    Install Python dependencies from requirements.txt into a target directory.

    Args:
        package_dir: Directory where packages will be installed
        requirements_path: Path to the requirements.txt file
        tool_key: Identifier for the tool (used for logging)
        toolkit_version: The version of the toolkit
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
                str(package_dir),
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


def build_lambda_function_file(tool_entrypoint_module: str, tool_entrypoint_class: str) -> str:
    """
    Build a Lambda function file from a template.

    Args:
        tool_entrypoint_module: The module name containing the entrypoint
        tool_entrypoint_class: The class name to use as entrypoint

    Returns:
        The contents of the Lambda function file
    """
    template_path = Path(__file__).parent / "templates" / "lambda_function.py.template"

    with open(template_path) as template_file:
        template_content = template_file.read()

    # Escape literal curly braces in the template by doubling them
    template_content = template_content.replace("{", "{{").replace("}", "}}")

    # Un-escape the placeholders we want to replace
    template_content = template_content.replace("{{module}}", "{module}").replace("{{class_name}}", "{class_name}")

    template_content = template_content.replace("{{sentry_dsn}}", settings.FUNCTION_SENTRY_DSN)

    # Replace placeholders in the template
    return template_content.format(module=tool_entrypoint_module, class_name=tool_entrypoint_class)


def create_tool_zip(  # noqa: PLR0913, PLR0915
    tool_folder_zip_content: bytes,
    tool_key: str,
    project_uuid: str,
    tool_entrypoint_module: str,
    tool_entrypoint_class: str,
    toolkit_version: str,
) -> BytesIO:
    """
    Create a tool zip file from a folder zip file.

    1. Extracts the tool folder zip content to a temporary directory
    2. Installs dependencies from requirements.txt if present
    3. Creates a Lambda function entry point
    4. Re-zips the content with installed dependencies

    Args:
        tool_folder_zip_content: Bytes content of the zipped tool folder
        tool_key: The tool key identifier
        project_uuid: The UUID of the project
        tool_entrypoint_module: The module name containing the entrypoint
        tool_entrypoint_class: The class name to use as entrypoint
        toolkit_version: The version of the toolkit
    Returns:
        BytesIO: A buffer containing the zipped tool with installed dependencies

    Raises:
        ValueError: If there are issues with the requirements file or installation
        IOError: If there are issues with file operations
    """
    temp_dir = None
    temp_prefix = f"tool_{tool_key}_{project_uuid}_"

    try:
        logger.info(f"Creating tool zip for {tool_key} in project {project_uuid}")

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix=temp_prefix)
        temp_path = Path(temp_dir)

        # Extract zip to temporary directory
        zip_buffer = BytesIO(tool_folder_zip_content)
        with zipfile.ZipFile(zip_buffer) as zip_ref:
            logger.debug(f"Extracting {len(zip_ref.namelist())} files to {temp_dir}")
            zip_ref.extractall(temp_dir)

        # Create package directory
        package_dir = temp_path / "package"
        package_dir.mkdir(exist_ok=True)

        # Install toolkit in package directory
        default_packages = [
            Package("weni-agents-toolkit", toolkit_version),
            Package("sentry-sdk", "2.24.1"),
        ]

        install_packages(package_dir, default_packages)

        # Check for requirements.txt
        requirements_path = temp_path / "requirements.txt"
        if requirements_path.exists():
            logger.info(f"Found requirements.txt in {tool_key}")

            # Validate requirements file for security
            is_valid, error_message = validate_requirements_file(str(requirements_path))
            if not is_valid:
                logger.warning(f"Invalid requirements.txt in {tool_key}: {error_message}")
                raise ValueError(f"Invalid requirements.txt: {error_message}")

            # Install dependencies
            install_dependencies(package_dir, requirements_path, tool_key, toolkit_version)

        # Create the lambda function file
        logger.info(f"Creating Lambda function for {tool_key}")
        lambda_function_content = build_lambda_function_file(tool_entrypoint_module, tool_entrypoint_class)
        lambda_function_path = temp_path / "lambda_function.py"
        with open(lambda_function_path, "w") as f:
            f.write(lambda_function_content)

        # Create a new zip
        logger.info(f"Creating output zip for {tool_key}")
        output_buffer = BytesIO()
        with zipfile.ZipFile(output_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:
            # Walk through all files in the temp directory
            for root, _, files in os.walk(temp_dir):
                root_path = Path(root)
                for file in files:
                    file_path = root_path / file
                    arc_name = file_path.relative_to(temp_path)

                    # Determine the path in the zip file
                    if str(arc_name) == "lambda_function.py":
                        # Keep these at the root
                        zip_path = str(arc_name)
                    elif str(arc_name).startswith("package/"):
                        # move content out of /package and insert it at root
                        zip_path = str(arc_name).replace("package/", "", 1)
                    else:
                        # Move other files to tool/ directory
                        zip_path = f"tool/{arc_name}"

                    zip_out.write(file_path, arcname=zip_path)

        # Prepare the output
        output_buffer.seek(0)
        output_buffer.name = f"{tool_key}.zip"
        logger.info(f"Completed creation of tool zip for {tool_key}")
        return output_buffer

    except Exception as e:
        logger.exception(f"Error creating tool zip for {tool_key}: {str(e)}")
        raise

    finally:
        # Clean up the temporary directory
        if temp_dir and Path(temp_dir).exists():
            logger.debug(f"Cleaning up temporary directory {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


async def process_tool(  # noqa: PLR0913
    folder_zip: bytes,
    key: str,
    project_uuid: str,
    agent_slug: str,
    tool_slug: str,
    definition: dict[str, Any],
    processed_count: int,
    total_count: int,
    toolkit_version: str,
) -> tuple[CLIResponse, BytesIO | None]:
    """
    Process a single tool file.

    Args:
        folder_zip: The tool folder zip file content
        key: The tool key (agent:tool)
        project_uuid: The UUID of the project
        agent_slug: The slug of the agent
        tool_slug: The name of the tool
        definition: The definition data
        processed_count: The number of tools processed so far
        total_count: The total number of tools to process
        toolkit_version: The version of the toolkit

    Returns:
        Tuple of (response, tool_zip_bytes or None if error)
    """
    # Reduce the progress to be always between 0.2 and 0.9
    progress = 0.2 + (processed_count / total_count) * 0.7
    logger.info(f"Processing tool {tool_slug} for agent {agent_slug} ({processed_count}/{total_count})")

    try:
        # Get tool entrypoint
        agent_info = next(
            (agent for agent in definition["agents"].values() if agent["slug"] == agent_slug),
            None,
        )

        if not agent_info:
            raise ValueError(f"Could not find agent {agent_slug} in definition")

        tool_info = next(
            (tool for tool in agent_info["tools"] if tool["slug"] == tool_slug),
            None,
        )

        if not tool_info:
            raise ValueError(f"Could not find tool {tool_slug} for agent {agent_slug} in definition")

        tool_entrypoint = tool_info["source"]["entrypoint"]
        tool_entrypoint_module = tool_entrypoint.split(".")[0]
        tool_entrypoint_class = tool_entrypoint.split(".")[1]
        logger.debug(f"Tool entrypoint: {tool_entrypoint_module}.{tool_entrypoint_class}")

        # Create zip package
        logger.info(f"Creating zip package for tool {tool_slug}")
        tool_zip_bytes = create_tool_zip(
            folder_zip,
            key,
            str(project_uuid),
            tool_entrypoint_module,
            tool_entrypoint_class,
            toolkit_version,
        )

        file_size_kb = len(tool_zip_bytes.getvalue()) / 1024
        logger.debug(f"Successfully created zip for {key}, size: {file_size_kb:.2f} KB")

        # Prepare success response
        response: CLIResponse = {
            "message": f"Tool {tool_slug} processed successfully ({processed_count}/{total_count})",
            "data": {
                "tool_name": tool_slug,
                "agent_name": agent_slug,
                "size_kb": round(file_size_kb, 2),
                "processed": processed_count,
                "total": total_count,
            },
            "success": True,
            "code": "TOOL_PROCESSED",
            "progress": progress,
        }

        return response, tool_zip_bytes

    except Exception as e:
        logger.error(f"Failed to process tool {key}: {str(e)}")
        error_response: CLIResponse = {
            "message": f"Failed to process tool {key}",
            "data": {
                "tool_name": tool_slug,
                "agent_name": agent_slug,
                "error": str(e),
                "processed": processed_count,
                "total": total_count,
            },
            "success": False,
            "code": "TOOL_PROCESSING_ERROR",
            "progress": progress,
        }
        return error_response, None
