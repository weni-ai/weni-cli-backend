"""
Utilities for packaging tools into deployable zip files.
"""

import logging
import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from app.core.response import CLIResponse
from app.services.package import Package, Packager
from app.services.tool.validator import validate_requirements_file

# Constants
logger = logging.getLogger(__name__)


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

        packager = Packager(package_dir)
        packager.install_packages(default_packages)

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
            packager.install_dependencies(requirements_path, tool_key)

        # Create the lambda function file
        logger.info(f"Creating Lambda function for {tool_key}")
        replacements = {
            "module": tool_entrypoint_module,
            "class_name": tool_entrypoint_class,
        }
        template_path = Path(__file__).parent / "templates" / "lambda_function.py.template"
        lambda_function_content = packager.build_lambda_function_file(template_path, replacements)
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
    agent_key: str,
    tool_key: str,
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
        agent_key: The key of the agent
        tool_key: The key of the tool
        definition: The definition data
        processed_count: The number of tools processed so far
        total_count: The total number of tools to process
        toolkit_version: The version of the toolkit

    Returns:
        Tuple of (response, tool_zip_bytes or None if error)
    """
    # Reduce the progress to be always between 0.2 and 0.9
    progress = 0.2 + (processed_count / total_count) * 0.7
    logger.info(f"Processing tool {tool_key} for agent {agent_key} ({processed_count}/{total_count})")

    try:
        # Get tool entrypoint
        agent_info = definition["agents"].get(agent_key)
        if not agent_info:
            raise ValueError(f"Could not find agent {agent_key} in definition")

        tool_info = next(
            (tool for tool in agent_info["tools"] if tool["key"] == tool_key),
            None,
        )

        if not tool_info:
            raise ValueError(f"Could not find tool {tool_key} for agent {agent_key} in definition")

        tool_entrypoint = tool_info["source"]["entrypoint"]
        tool_entrypoint_module = tool_entrypoint.split(".")[0]
        tool_entrypoint_class = tool_entrypoint.split(".")[1]
        logger.debug(f"Tool entrypoint: {tool_entrypoint_module}.{tool_entrypoint_class}")

        # Create zip package
        logger.info(f"Creating zip package for tool {tool_key}")
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
            "message": f"Tool {tool_key} processed successfully ({processed_count}/{total_count})",
            "data": {
                "tool_key": tool_key,
                "agent_key": agent_key,
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
                "tool_key": tool_key,
                "agent_key": agent_key,
                "error": str(e),
                "processed": processed_count,
                "total": total_count,
            },
            "success": False,
            "code": "TOOL_PROCESSING_ERROR",
            "progress": progress,
        }
        return error_response, None
