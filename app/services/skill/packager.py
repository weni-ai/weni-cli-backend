"""
Utilities for packaging skills into deployable zip files.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from app.services.skill.validator import validate_requirements_file

# Constants
SUBPROCESS_TIMEOUT_SECONDS = 120
logger = logging.getLogger(__name__)


def install_toolkit(package_dir: Path, toolkit_version: str) -> None:
    """
    Install the weni-agents-toolkit into a target directory.
    """
    try:
        process = subprocess.run(
            [
                "pip",
                "install",
                "--target",
                str(package_dir),
                f"weni-agents-toolkit=={toolkit_version}",
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


def install_dependencies(package_dir: Path, requirements_path: Path, skill_key: str, toolkit_version: str) -> None:
    """
    Install Python dependencies from requirements.txt into a target directory.

    Args:
        package_dir: Directory where packages will be installed
        requirements_path: Path to the requirements.txt file
        skill_key: Identifier for the skill (used for logging)
        toolkit_version: The version of the toolkit
    Raises:
        ValueError: If installation fails or times out
    """
    logger.info(f"Installing dependencies for {skill_key}")

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
        logger.error(f"Timeout installing requirements for {skill_key}")
        raise ValueError(f"Timeout installing requirements (>{SUBPROCESS_TIMEOUT_SECONDS}s)") from e


def build_lambda_function_file(skill_entrypoint_module: str, skill_entrypoint_class: str) -> str:
    """
    Build a Lambda function file from a template.

    Args:
        skill_entrypoint_module: The module name containing the entrypoint
        skill_entrypoint_class: The class name to use as entrypoint

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

    # Replace placeholders in the template
    return template_content.format(module=skill_entrypoint_module, class_name=skill_entrypoint_class)


def create_skill_zip(  # noqa: PLR0913
    skill_folder_zip_content: bytes,
    skill_key: str,
    project_uuid: str,
    skill_entrypoint_module: str,
    skill_entrypoint_class: str,
    toolkit_version: str,
) -> BytesIO:
    """
    Create a skill zip file from a folder zip file.

    1. Extracts the skill folder zip content to a temporary directory
    2. Installs dependencies from requirements.txt if present
    3. Creates a Lambda function entry point
    4. Re-zips the content with installed dependencies

    Args:
        skill_folder_zip_content: Bytes content of the zipped skill folder
        skill_key: The skill key identifier
        project_uuid: The UUID of the project
        skill_entrypoint_module: The module name containing the entrypoint
        skill_entrypoint_class: The class name to use as entrypoint
        toolkit_version: The version of the toolkit
    Returns:
        BytesIO: A buffer containing the zipped skill with installed dependencies

    Raises:
        ValueError: If there are issues with the requirements file or installation
        IOError: If there are issues with file operations
    """
    temp_dir = None
    temp_prefix = f"skill_{skill_key}_{project_uuid}_"

    try:
        logger.info(f"Creating skill zip for {skill_key} in project {project_uuid}")

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix=temp_prefix)
        temp_path = Path(temp_dir)

        # Extract zip to temporary directory
        zip_buffer = BytesIO(skill_folder_zip_content)
        with zipfile.ZipFile(zip_buffer) as zip_ref:
            logger.debug(f"Extracting {len(zip_ref.namelist())} files to {temp_dir}")
            zip_ref.extractall(temp_dir)

        # Create package directory
        package_dir = temp_path / "package"
        package_dir.mkdir(exist_ok=True)

        # Install toolkit in package directory
        install_toolkit(package_dir, toolkit_version)

        # Check for requirements.txt
        requirements_path = temp_path / "requirements.txt"
        if requirements_path.exists():
            logger.info(f"Found requirements.txt in {skill_key}")

            # Validate requirements file for security
            is_valid, error_message = validate_requirements_file(str(requirements_path))
            if not is_valid:
                logger.warning(f"Invalid requirements.txt in {skill_key}: {error_message}")
                raise ValueError(f"Invalid requirements.txt: {error_message}")

            # Install dependencies
            install_dependencies(package_dir, requirements_path, skill_key, toolkit_version)

        # Create the lambda function file
        logger.info(f"Creating Lambda function for {skill_key}")
        lambda_function_content = build_lambda_function_file(skill_entrypoint_module, skill_entrypoint_class)
        lambda_function_path = temp_path / "lambda_function.py"
        with open(lambda_function_path, "w") as f:
            f.write(lambda_function_content)

        # Create a new zip
        logger.info(f"Creating output zip for {skill_key}")
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
                        # Move other files to skill/ directory
                        zip_path = f"skill/{arc_name}"

                    zip_out.write(file_path, arcname=zip_path)

        # Prepare the output
        output_buffer.seek(0)
        output_buffer.name = f"{skill_key}.zip"
        logger.info(f"Completed creation of skill zip for {skill_key}")
        return output_buffer

    except Exception as e:
        logger.exception(f"Error creating skill zip for {skill_key}: {str(e)}")
        raise

    finally:
        # Clean up the temporary directory
        if temp_dir and Path(temp_dir).exists():
            logger.debug(f"Cleaning up temporary directory {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
