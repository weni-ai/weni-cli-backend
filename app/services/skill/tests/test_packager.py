"""
Tests for the skill package packager.
"""

import io
import os
import subprocess
import zipfile
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, ClassVar
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from app.services.skill.packager import (
    SUBPROCESS_TIMEOUT_SECONDS,
    Package,
    build_lambda_function_file,
    create_skill_zip,
    install_dependencies,
    install_packages,
)

# Common test constants
TEST_CONTENT = b"test content"
TEST_AGENT = "test-agent"
TEST_SKILL = "test-skill"
TEST_SKILL_KEY = f"{TEST_AGENT}:{TEST_SKILL}"
TEST_TOKEN = "Bearer test-token"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Creates a temporary directory for testing."""
    with TemporaryDirectory() as temp_dir_path:
        yield Path(temp_dir_path)


@pytest.fixture
def requirements_file() -> Generator[Path, None, None]:
    """Creates a temporary requirements file for testing."""
    with NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"pytest==7.3.1\nrequests==2.31.0\n")
        temp_file.flush()
        yield Path(temp_file.name)
        # Clean up after test
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@pytest.fixture
def skill_folder_zip() -> BytesIO:
    """Creates a sample skill folder zip for testing."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        # Add a sample Python file
        zip_file.writestr(
            "skill.py",
            "class SkillHandler:\n    def handle(self, event, context):\n        return {'message': 'Hello'}",
        )
        # Add a requirements.txt file
        zip_file.writestr("requirements.txt", "requests==2.31.0\npydantic>=2.0.0\n")
        # Add a README file
        zip_file.writestr("README.md", "# Sample Skill\nThis is a sample skill.")

    buffer.seek(0)
    return buffer


class TestInstallToolkit:
    """Tests for the install_packages function."""

    def test_successful_installation(self, temp_dir: Path, mocker: MockerFixture) -> None:
        """Test successful toolkit installation."""
        # Mock subprocess.run to avoid actual installation
        mock_process = mocker.Mock()
        mock_process.stdout = "Successfully installed weni-agents-toolkit-1.0.0"
        mock_run = mocker.Mock(return_value=mock_process)
        mocker.patch("subprocess.run", mock_run)

        packages = [Package("weni-agents-toolkit", "1.0.0")]

        # Call the function
        install_packages(temp_dir, packages)

        # Verify subprocess.run was called with correct arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "pip" in call_args[0], "Should call pip"
        assert "install" in call_args[1], "Should use install command"
        assert "weni-agents-toolkit==1.0.0" in call_args[4], "Should specify toolkit version"

        # Check timer option
        timeout_arg = mock_run.call_args[1].get("timeout")
        assert timeout_arg == SUBPROCESS_TIMEOUT_SECONDS, "Should set timeout"

    def test_installation_failure(self, temp_dir: Path, mocker: MockerFixture) -> None:
        """Test toolkit installation failure."""
        # Mock subprocess.run to raise CalledProcessError
        error = subprocess.CalledProcessError(1, ["pip", "install"], stderr="Could not find package")
        mock_run = mocker.Mock(side_effect=error)
        mocker.patch("subprocess.run", mock_run)

        packages = [Package("weni-agents-toolkit", "1.0.0")]

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            install_packages(temp_dir, packages)
        assert "Failed to install toolkit" in str(exc_info.value), "Error message should be descriptive"
        assert "Could not find package" in str(exc_info.value), "Should include the subprocess error"

    def test_installation_timeout(self, temp_dir: Path, mocker: MockerFixture) -> None:
        """Test toolkit installation timeout."""
        # Mock subprocess.run to raise TimeoutExpired
        error = subprocess.TimeoutExpired(["pip", "install"], SUBPROCESS_TIMEOUT_SECONDS)
        mock_run = mocker.Mock(side_effect=error)
        mocker.patch("subprocess.run", mock_run)

        packages = [Package("weni-agents-toolkit", "1.0.0")]

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            install_packages(temp_dir, packages)
        assert "Timeout installing toolkit" in str(exc_info.value), "Error should mention timeout"
        assert str(SUBPROCESS_TIMEOUT_SECONDS) in str(exc_info.value), "Should include timeout duration"


class TestInstallDependencies:
    """Tests for the install_dependencies function."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for testing."""
        with TemporaryDirectory() as temp_dir_path:
            yield Path(temp_dir_path)

    @pytest.fixture
    def requirements_file(self) -> Generator[Path, None, None]:
        """Create temporary requirements file for testing."""
        with NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(b"pytest==7.3.1\nrequests==2.31.0\n")
            temp_file.flush()
            yield Path(temp_file.name)
            # Clean up after test
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def test_successful_installation(self, temp_dir: Path, requirements_file: Path, mocker: MockerFixture) -> None:
        """Test successful dependency installation."""
        # Mock subprocess.run to avoid actual installation
        mock_process = mocker.Mock()
        mock_process.stdout = "Successfully installed pytest-7.3.1 requests-2.31.0"
        mock_run = mocker.Mock(return_value=mock_process)
        mocker.patch("subprocess.run", mock_run)

        # Call the function
        install_dependencies(temp_dir, requirements_file, "test-skill", "1.0.0")

        # Verify subprocess.run was called with correct arguments - now called once for regular dependencies
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "pip" in call_args[0], "Should call pip"
        assert "install" in call_args[1], "Should use install command"
        assert str(requirements_file) in call_args[5], "Should specify requirements file"

        # Check timer option
        timeout_arg = mock_run.call_args[1].get("timeout")
        assert timeout_arg == SUBPROCESS_TIMEOUT_SECONDS, "Should set timeout"

    def test_installation_failure(self, temp_dir: Path, requirements_file: Path, mocker: MockerFixture) -> None:
        """Test dependency installation failure."""
        # Mock subprocess.run to raise CalledProcessError
        error = subprocess.CalledProcessError(1, ["pip", "install"], stderr="Could not find package")
        mock_run = mocker.Mock(side_effect=error)
        mocker.patch("subprocess.run", mock_run)

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            install_dependencies(temp_dir, requirements_file, "test-skill", "1.0.0")
        assert "Failed to install requirements" in str(exc_info.value), "Error message should be descriptive"
        assert "Could not find package" in str(exc_info.value), "Should include the subprocess error"

    def test_installation_timeout(self, temp_dir: Path, requirements_file: Path, mocker: MockerFixture) -> None:
        """Test dependency installation timeout."""
        # Mock subprocess.run to raise TimeoutExpired
        error = subprocess.TimeoutExpired(["pip", "install"], SUBPROCESS_TIMEOUT_SECONDS)
        mock_run = mocker.Mock(side_effect=error)
        mocker.patch("subprocess.run", mock_run)

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            install_dependencies(temp_dir, requirements_file, "test-skill", "1.0.0")
        assert "Timeout installing requirements" in str(exc_info.value), "Error should mention timeout"
        assert str(SUBPROCESS_TIMEOUT_SECONDS) in str(exc_info.value), "Should include timeout duration"


class TestBuildLambdaFunctionFile:
    """Tests for the build_lambda_function_file function."""

    def test_successful_template_creation(self) -> None:
        """Test successful Lambda function template creation."""
        module_name = "skill_module"
        class_name = "SkillHandler"

        # Call the function
        result = build_lambda_function_file(module_name, class_name)

        # Verify specific content is correctly substituted
        assert "import json" in result
        assert "from weni.context import Context" in result
        assert "from skill.skill_module import SkillHandler" in result
        assert "def lambda_handler(event, context):" in result
        assert "result, format = SkillHandler(context)" in result
        assert "dummy_function_response = {'response': action_response, 'messageVersion': '1.0'}" in result
        assert "sentry_sdk.init(" in result

    def test_template_substitution(self) -> None:
        """Test template variable substitution with different values."""
        result = build_lambda_function_file("custom_module", "CustomClass")

        assert "from skill.custom_module import CustomClass" in result
        assert "result, format = CustomClass(context)" in result
        # Ensure other parts of the template are intact
        assert "json.loads(session_attributes.get('credentials'))" in result
        assert "promptSessionAttributes" in result

    def test_template_not_found(self, mocker: MockerFixture) -> None:
        """Test behavior when template file is not found."""
        # Make open raise FileNotFoundError
        mocker.patch("builtins.open", side_effect=FileNotFoundError("Template not found"))

        with pytest.raises(FileNotFoundError):
            build_lambda_function_file("module", "Class")


class TestCreateSkillZip:
    """Tests for the create_skill_zip function."""

    @pytest.fixture
    def skill_folder_zip(self) -> BytesIO:
        """Create a test zip file containing a skill."""
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zip_file:
            zip_file.writestr("skill.py", "class SkillHandler: pass")
            zip_file.writestr("requirements.txt", "pytest==7.3.1\nrequests==2.31.0\n")
        buffer.seek(0)
        return buffer

    def test_successful_zip_creation(self, skill_folder_zip: BytesIO, mocker: MockerFixture) -> None:
        """Test successful creation of skill zip."""
        # Mock validation to return success
        mocker.patch("app.services.skill.packager.validate_requirements_file", return_value=(True, ""))

        # Mock toolkit installation
        mock_install_packages = mocker.patch("app.services.skill.packager.install_packages")

        # Mock dependencies installation
        mock_install_dependencies = mocker.patch("app.services.skill.packager.install_dependencies")

        # Mock lambda function file creation
        mocker.patch(
            "app.services.skill.packager.build_lambda_function_file",
            return_value="def lambda_handler(event, context): pass",
        )

        # Call the function
        result = create_skill_zip(
            skill_folder_zip.getvalue(), "test-skill", "project-123", "skill_module", "SkillHandler", "1.0.0"
        )

        # Verify result
        assert result is not None, "Should return a file-like object"
        assert isinstance(result, BytesIO), "Should return BytesIO"

        # Verify toolkit and dependencies were installed
        mock_install_packages.assert_called_once()
        mock_install_dependencies.assert_called_once()

        # Check that the result is a valid zip file
        with zipfile.ZipFile(result) as zip_file:
            assert "lambda_function.py" in zip_file.namelist(), "Should include lambda function"

    def test_invalid_requirements(self, skill_folder_zip: BytesIO, mocker: MockerFixture) -> None:
        """Test zip creation with invalid requirements."""
        # Mock toolkit installation
        mocker.patch("app.services.skill.packager.install_packages")

        # Mock validation to return failure
        mocker.patch(
            "app.services.skill.packager.validate_requirements_file", return_value=(False, "Invalid requirements file")
        )

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            create_skill_zip(
                skill_folder_zip.getvalue(), "test-skill", "project-123", "skill_module", "SkillHandler", "1.0.0"
            )
        assert "Invalid requirements file" in str(exc_info.value), "Should include validation message"

    def test_installation_error(self, skill_folder_zip: BytesIO, mocker: MockerFixture) -> None:
        """Test zip creation with dependency installation error."""
        # Mock toolkit installation
        mocker.patch("app.services.skill.packager.install_packages")

        # Mock validation to return success
        mocker.patch("app.services.skill.packager.validate_requirements_file", return_value=(True, ""))

        # Mock install_dependencies to raise ValueError
        mocker.patch("app.services.skill.packager.install_dependencies", side_effect=ValueError("Installation failed"))

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            create_skill_zip(
                skill_folder_zip.getvalue(), "test-skill", "project-123", "skill_module", "SkillHandler", "1.0.0"
            )
        assert "Installation failed" in str(exc_info.value), "Should propagate the error message"

    def test_toolkit_installation_error(self, skill_folder_zip: BytesIO, mocker: MockerFixture) -> None:
        """Test zip creation with toolkit installation error."""
        # Mock install_packages to raise ValueError
        mocker.patch(
            "app.services.skill.packager.install_packages", side_effect=ValueError("Toolkit installation failed")
        )

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            create_skill_zip(
                skill_folder_zip.getvalue(), "test-skill", "project-123", "skill_module", "SkillHandler", "1.0.0"
            )
        assert "Toolkit installation failed" in str(exc_info.value), "Should propagate the toolkit error"

    def test_invalid_zip_content(self) -> None:
        """Test with invalid zip content."""
        # Create invalid zip content
        invalid_content = b"not a valid zip file"

        # Verify function raises zipfile.BadZipFile
        with pytest.raises(zipfile.BadZipFile):
            create_skill_zip(invalid_content, "test-skill", "project-123", "skill_module", "SkillHandler", "1.0.0")

    def test_no_requirements_file(self, mocker: MockerFixture) -> None:
        """Test zip creation with no requirements.txt file."""
        # Create a zip without requirements.txt
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zip_file:
            zip_file.writestr("skill.py", "class SkillHandler: pass")

        # Mock toolkit installation
        mock_install_packages = mocker.patch("app.services.skill.packager.install_packages")

        # Mock dependencies installation - should NOT be called
        mock_install_dependencies = mocker.patch("app.services.skill.packager.install_dependencies")

        # Mock lambda function file creation
        mocker.patch(
            "app.services.skill.packager.build_lambda_function_file",
            return_value="def lambda_handler(event, context): pass",
        )

        buffer.seek(0)

        # Call the function with toolkit_version parameter
        result = create_skill_zip(
            buffer.getvalue(), "test-skill", "project-123", "skill_module", "SkillHandler", "1.0.0"
        )

        # Verify result
        assert result is not None, "Should return a file-like object"
        assert isinstance(result, BytesIO), "Should return BytesIO"

        # Verify toolkit was installed but dependencies were not
        mock_install_packages.assert_called_once()
        mock_install_dependencies.assert_not_called()

        with zipfile.ZipFile(result) as zip_file:
            assert "lambda_function.py" in zip_file.namelist(), "Should include lambda function"

    def test_cleanup_on_exception(self, mocker: MockerFixture) -> None:
        """Test that temporary directory is cleaned up on exception."""
        # Mock tempfile.mkdtemp to return a fixed path
        temp_dir_path = "/tmp/test_skill_dir"
        mocker.patch("app.services.skill.packager.tempfile.mkdtemp", return_value=temp_dir_path)

        # Mock Path.exists to return True when checking the temp directory
        mocker.patch("app.services.skill.packager.Path.exists", return_value=True)

        # Mock shutil.rmtree to verify it's called
        rmtree_mock = mocker.Mock()
        mocker.patch("app.services.skill.packager.shutil.rmtree", rmtree_mock)

        # Create invalid zip content to force an exception
        invalid_content = b"not a valid zip file"

        # Call the function and catch the exception
        with pytest.raises(zipfile.BadZipFile):
            create_skill_zip(invalid_content, "test-skill", "project-123", "skill_module", "SkillHandler", "1.0.0")

        # Verify cleanup was attempted (now includes ignore_errors=True)
        rmtree_mock.assert_called_once_with(temp_dir_path, ignore_errors=True), "Should clean up temp directory"


class TestProcessSkill:
    """Tests for process_skill function."""

    # Common test data
    folder_zip: ClassVar[bytes] = b"test zip content"
    key: ClassVar[str] = TEST_SKILL_KEY
    project_uuid: ClassVar[str] = str(uuid4())
    agent_name: ClassVar[str] = TEST_AGENT
    skill_name: ClassVar[str] = TEST_SKILL
    skill_definition: ClassVar[dict[str, Any]] = {
        "agents": {
            TEST_AGENT: {
                "slug": TEST_AGENT,
                "skills": [
                    {
                        "slug": TEST_SKILL,
                        "source": {"entrypoint": "main.TestSkill"},
                    }
                ],
            }
        }
    }
    processed_count: ClassVar[int] = 1
    total_count: ClassVar[int] = 2
    expected_progress: ClassVar[float] = 0.55  # 1/2 * 0.7 + 0.2
    toolkit_version: ClassVar[str] = "1.0.0"  # Add default toolkit version

    def test_success(self) -> None:
        """Test successful processing of a skill."""
        import asyncio

        from app.services.skill.packager import process_skill

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.services.skill.packager.create_skill_zip", lambda *args, **kwargs: io.BytesIO(b"packaged content")
            )

            response, skill_zip = asyncio.run(
                process_skill(
                    self.folder_zip,
                    self.key,
                    self.project_uuid,
                    self.agent_name,
                    self.skill_name,
                    self.skill_definition,
                    self.processed_count,
                    self.total_count,
                    self.toolkit_version,  # Add toolkit version parameter
                )
            )

        assert response["success"] is True, "Should report success"
        assert skill_zip is not None, "Should return skill zip"
        assert response["data"] is not None, "Response should include data"
        assert response["data"]["skill_name"] == self.skill_name, "Should include skill name"
        assert response["progress"] == self.expected_progress, f"Progress should be {self.expected_progress}"

    def test_missing_agent_in_definition(self) -> None:
        """Test handling of missing agent in definition."""
        import asyncio

        from app.services.skill.packager import process_skill

        # Use a non-existent agent name
        missing_agent = "non-existent-agent"

        response, skill_zip = asyncio.run(
            process_skill(
                self.folder_zip,
                f"{missing_agent}:{self.skill_name}",
                self.project_uuid,
                missing_agent,  # Use the missing agent name
                self.skill_name,
                self.skill_definition,  # Definition only contains TEST_AGENT
                self.processed_count,
                self.total_count,
                self.toolkit_version,  # Add toolkit version parameter
            )
        )

        assert response["success"] is False, "Should report failure"
        assert skill_zip is None, "Should not return skill zip"
        assert response["data"] is not None, "Response should include data"
        assert "Could not find agent" in response["data"]["error"], "Error should mention missing agent"
        assert missing_agent in response["data"]["error"], "Error should include the missing agent name"

    @pytest.mark.parametrize(
        "scenario, key, skill_name, expected_error",
        [
            ("missing_skill", "test-agent:missing-skill", "missing-skill", "Could not find skill"),
            ("exception", TEST_SKILL_KEY, TEST_SKILL, "Zip creation error"),
        ],
    )
    def test_error_scenarios(self, scenario: str, key: str, skill_name: str, expected_error: str) -> None:
        """Test various error scenarios in skill processing."""
        import asyncio

        from app.api.v1.routers.agents import process_skill

        with pytest.MonkeyPatch().context() as mp:
            if scenario == "exception":
                # Cause an exception
                mp.setattr(
                    "app.services.skill.packager.create_skill_zip",
                    lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Zip creation error")),
                )

            response, skill_zip = asyncio.run(
                process_skill(
                    self.folder_zip,
                    key,
                    self.project_uuid,
                    self.agent_name,
                    skill_name,
                    self.skill_definition,
                    self.processed_count,
                    self.total_count,
                    self.toolkit_version,  # Add toolkit version parameter
                )
            )

        assert response["success"] is False, "Should report failure"
        assert response["data"] is not None, "Response should include data"
        assert expected_error in response["data"]["error"], f"Should include '{expected_error}' in error message"
        assert skill_zip is None, "Should not return skill zip"
