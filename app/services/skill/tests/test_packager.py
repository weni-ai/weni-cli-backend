"""
Tests for the skill package packager.
"""

import os
import subprocess
import zipfile
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest
from pytest_mock import MockerFixture

from app.services.skill.packager import (
    SUBPROCESS_TIMEOUT_SECONDS,
    build_lambda_function_file,
    create_skill_zip,
    install_dependencies,
)


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


class TestInstallDependencies:
    """Tests for the install_dependencies function."""

    def test_successful_installation(self, temp_dir: Path, requirements_file: Path, mocker: MockerFixture) -> None:
        """Test successful dependency installation."""
        # Mock subprocess.run to avoid actual installation
        mock_process = mocker.Mock()
        mock_process.stdout = "Successfully installed pytest-7.3.1 requests-2.31.0"
        mock_run = mocker.Mock(return_value=mock_process)
        mocker.patch("subprocess.run", mock_run)

        # Call the function
        install_dependencies(temp_dir, requirements_file, "test-skill")

        # Verify subprocess.run was called with correct arguments
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0][0] == "pip"
        assert args[0][1] == "install"
        assert args[0][2] == "--target"
        assert args[0][3] == str(temp_dir)
        assert args[0][4] == "-r"
        assert args[0][5] == str(requirements_file)
        assert "timeout" in kwargs
        assert kwargs["timeout"] == SUBPROCESS_TIMEOUT_SECONDS

    def test_installation_failure(self, temp_dir: Path, requirements_file: Path, mocker: MockerFixture) -> None:
        """Test dependency installation failure."""
        # Mock subprocess.run to raise CalledProcessError
        error = subprocess.CalledProcessError(1, ["pip", "install"], stderr="Could not find package")
        mock_run = mocker.Mock(side_effect=error)
        mocker.patch("subprocess.run", mock_run)

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            install_dependencies(temp_dir, requirements_file, "test-skill")

        assert "Failed to install requirements" in str(exc_info.value)

    def test_installation_timeout(self, temp_dir: Path, requirements_file: Path, mocker: MockerFixture) -> None:
        """Test dependency installation timeout."""
        # Mock subprocess.run to raise TimeoutExpired
        error = subprocess.TimeoutExpired(["pip", "install"], SUBPROCESS_TIMEOUT_SECONDS)
        mock_run = mocker.Mock(side_effect=error)
        mocker.patch("subprocess.run", mock_run)

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            install_dependencies(temp_dir, requirements_file, "test-skill")

        assert "Timeout installing requirements" in str(exc_info.value)
        assert str(SUBPROCESS_TIMEOUT_SECONDS) in str(exc_info.value)


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

    def test_successful_zip_creation(self, skill_folder_zip: BytesIO, mocker: MockerFixture) -> None:
        """Test successful creation of skill zip."""
        # Mock validation to return success
        mocker.patch("app.services.skill.packager.validate_requirements_file", return_value=(True, ""))

        # Mock dependencies installation
        mocker.patch("app.services.skill.packager.install_dependencies")

        # Mock lambda function file creation
        mocker.patch(
            "app.services.skill.packager.build_lambda_function_file",
            return_value="def lambda_handler(event, context): pass",
        )

        # Call the function
        result = create_skill_zip(
            skill_folder_zip.getvalue(), "test-skill", "project-123", "skill_module", "SkillHandler"
        )

        # Verify result is a BytesIO object
        assert isinstance(result, BytesIO)

        # Verify the zip file is valid and contains expected entries
        with zipfile.ZipFile(result) as zip_out:
            file_list = zip_out.namelist()
            assert "lambda_function.py" in file_list
            assert any(f.startswith("skill/") for f in file_list)

    def test_invalid_requirements(self, skill_folder_zip: BytesIO, mocker: MockerFixture) -> None:
        """Test zip creation with invalid requirements."""
        # Mock validation to return failure
        mocker.patch(
            "app.services.skill.packager.validate_requirements_file", return_value=(False, "Invalid requirements file")
        )

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            create_skill_zip(skill_folder_zip.getvalue(), "test-skill", "project-123", "skill_module", "SkillHandler")

        assert "Invalid requirements.txt" in str(exc_info.value)

    def test_installation_error(self, skill_folder_zip: BytesIO, mocker: MockerFixture) -> None:
        """Test zip creation with dependency installation error."""
        # Mock validation to return success
        mocker.patch("app.services.skill.packager.validate_requirements_file", return_value=(True, ""))

        # Mock install_dependencies to raise ValueError
        mocker.patch("app.services.skill.packager.install_dependencies", side_effect=ValueError("Installation failed"))

        # Verify function raises ValueError
        with pytest.raises(ValueError) as exc_info:
            create_skill_zip(skill_folder_zip.getvalue(), "test-skill", "project-123", "skill_module", "SkillHandler")

        assert "Installation failed" in str(exc_info.value)

    def test_invalid_zip_content(self) -> None:
        """Test with invalid zip content."""
        # Create invalid zip content
        invalid_content = b"not a valid zip file"

        # Verify function raises zipfile.BadZipFile
        with pytest.raises(zipfile.BadZipFile):
            create_skill_zip(invalid_content, "test-skill", "project-123", "skill_module", "SkillHandler")

    def test_no_requirements_file(self, mocker: MockerFixture) -> None:
        """Test zip creation with no requirements.txt file."""
        # Create a zip without requirements.txt
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zip_file:
            zip_file.writestr("skill.py", "class SkillHandler: pass")
        buffer.seek(0)

        # Create mocks
        validate_mock = mocker.Mock()
        install_mock = mocker.Mock()

        # Mock lambda function file creation
        mocker.patch(
            "app.services.skill.packager.build_lambda_function_file",
            return_value="def lambda_handler(event, context): pass",
        )

        # Set the mocks to be checked later
        mocker.patch("app.services.skill.packager.validate_requirements_file", validate_mock)
        mocker.patch("app.services.skill.packager.install_dependencies", install_mock)

        # Call the function
        result = create_skill_zip(buffer.getvalue(), "test-skill", "project-123", "skill_module", "SkillHandler")

        # Verify mocks were not called
        assert validate_mock.call_count == 0
        assert install_mock.call_count == 0

        # Verify result is a BytesIO object with expected content
        assert isinstance(result, BytesIO)
        with zipfile.ZipFile(result) as zip_out:
            assert "lambda_function.py" in zip_out.namelist()

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
            create_skill_zip(invalid_content, "test-skill", "project-123", "skill_module", "SkillHandler")

        # Verify rmtree was called to clean up the directory
        rmtree_mock.assert_called_once_with(temp_dir_path, ignore_errors=True)
