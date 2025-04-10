"""
Tests for the tool package validator.
"""

import os
from collections.abc import Callable, Generator
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from pytest_mock import MockerFixture

from app.services.tool.validator import (
    DANGEROUS_PATTERNS,
    MAX_FILE_SIZE_MB,
    MAX_REQUIREMENTS_COUNT,
    UNSAFE_PATTERNS,
    _check_file_exists,
    _check_file_size,
    _check_for_dangerous_patterns,
    _check_for_unsafe_patterns,
    _check_requirements_count,
    validate_requirements_file,
)

type ValidationResult = tuple[bool, str]
type ValidationFunction = Callable[[list[str]], ValidationResult]


@pytest.fixture
def temp_requirements_file() -> Generator[Path, None, None]:
    """Creates a temporary file for testing."""
    with NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        yield Path(temp_file.name)
        # Clean up after test
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@pytest.fixture
def valid_requirements() -> list[str]:
    """Sample valid requirements."""
    return [
        "# This is a comment\n",
        "fastapi==0.100.0\n",
        "pydantic>=2.0.0\n",
        "pytest==7.3.1\n",
        "\n",
        "requests==2.31.0\n",
    ]


class TestFileExistsCheck:
    """Tests for the file existence check function."""

    def test_file_exists(self, temp_requirements_file: Path) -> None:
        """Test when file exists."""
        result, message = _check_file_exists(temp_requirements_file)
        assert result is True
        assert message == ""

    def test_file_does_not_exist(self) -> None:
        """Test when file does not exist."""
        non_existent_file = Path("/path/to/nonexistent/requirements.txt")
        result, message = _check_file_exists(non_existent_file)
        assert result is False
        assert "not found" in message


class TestFileSizeCheck:
    """Tests for the file size check function."""

    def test_file_size_within_limit(self, temp_requirements_file: Path) -> None:
        """Test when file size is within the limit."""
        # Write a small amount of data
        with open(temp_requirements_file, "w") as f:
            f.write("small content")

        result, message = _check_file_size(temp_requirements_file)
        assert result is True
        assert message == ""

    def test_file_size_exceeds_limit(self, temp_requirements_file: Path, mocker: MockerFixture) -> None:
        """Test when file size exceeds the limit."""

        # Mock the file size check instead of creating a large file
        class MockStat:
            def __init__(self) -> None:
                # Set size to exceed the limit (in bytes)
                self.st_size = (MAX_FILE_SIZE_MB + 10) * 1024 * 1024

        # Mock the stat method
        mocker.patch.object(Path, "stat", return_value=MockStat())

        result, message = _check_file_size(temp_requirements_file)
        assert result is False
        assert "Requirements file too large" in message
        assert f"max: {MAX_FILE_SIZE_MB}MB" in message


class TestRequirementsCountCheck:
    """Tests for the requirements count check function."""

    def test_requirements_count_within_limit(self, valid_requirements: list[str]) -> None:
        """Test when requirements count is within the limit."""
        result, message = _check_requirements_count(valid_requirements)
        assert result is True
        assert message == ""

    def test_empty_requirements(self) -> None:
        """Test with empty requirements list."""
        result, message = _check_requirements_count([])
        assert result is True
        assert message == ""

    def test_requirements_count_exceeds_limit(self) -> None:
        """Test when requirements count exceeds the limit."""
        # Create list with more requirements than allowed
        many_requirements = [f"package{i}==1.0.0\n" for i in range(MAX_REQUIREMENTS_COUNT + 10)]

        result, message = _check_requirements_count(many_requirements)
        assert result is False
        assert "Too many packages" in message
        assert f"max: {MAX_REQUIREMENTS_COUNT}" in message

    def test_ignores_comments_and_empty_lines(self) -> None:
        """Test that comments and empty lines are ignored in count."""
        requirements = [
            "# Comment 1\n",
            "\n",
            "package1==1.0.0\n",
            "# Comment 2\n",
            "package2==2.0.0\n",
            "\n",
        ]

        result, message = _check_requirements_count(requirements)
        assert result is True
        assert message == ""


class TestDangerousPatternsCheck:
    """Tests for the dangerous patterns check function."""

    def test_no_dangerous_patterns(self, valid_requirements: list[str]) -> None:
        """Test when no dangerous patterns are present."""
        result, message = _check_for_dangerous_patterns(valid_requirements)
        assert result is True
        assert message == ""

    @pytest.mark.parametrize("pattern", DANGEROUS_PATTERNS)
    def test_with_dangerous_patterns(self, pattern: str) -> None:
        """Test each dangerous pattern."""
        requirements = [
            "safe-package==1.0.0\n",
            f"unsafe-{pattern}1.0.0\n",
            "another-safe==2.0.0\n",
        ]

        result, message = _check_for_dangerous_patterns(requirements)
        assert result is False
        assert "unsafe" in message

    def test_ignores_comments(self) -> None:
        """Test that commented dangerous patterns are ignored."""
        requirements = [
            "# This is setuptools== in a comment\n",
            "safe-package==1.0.0\n",
        ]

        result, message = _check_for_dangerous_patterns(requirements)
        assert result is True
        assert message == ""


class TestUnsafePatternsCheck:
    """Tests for the unsafe patterns check function."""

    def test_no_unsafe_patterns(self, valid_requirements: list[str]) -> None:
        """Test when no unsafe patterns are present."""
        result, message = _check_for_unsafe_patterns(valid_requirements)
        assert result is True
        assert message == ""

    @pytest.mark.parametrize("pattern", UNSAFE_PATTERNS)
    def test_with_unsafe_patterns(self, pattern: str) -> None:
        """Test each unsafe pattern."""
        requirements = [
            "safe-package==1.0.0\n",
            f"something-with-{pattern}in-it\n",
            "another-safe==2.0.0\n",
        ]

        result, message = _check_for_unsafe_patterns(requirements)
        assert result is False
        assert "Unsupported installation method" in message

    def test_ignores_comments(self) -> None:
        """Test that commented unsafe patterns are ignored."""
        requirements = [
            "# This has git+ in a comment\n",
            "safe-package==1.0.0\n",
        ]

        result, message = _check_for_unsafe_patterns(requirements)
        assert result is True
        assert message == ""


class TestValidateRequirementsFile:
    """Tests for the main validate_requirements_file function."""

    def test_valid_requirements_file(self, temp_requirements_file: Path, valid_requirements: list[str]) -> None:
        """Test a valid requirements file."""
        with open(temp_requirements_file, "w") as f:
            f.writelines(valid_requirements)

        result, message = validate_requirements_file(str(temp_requirements_file))
        assert result is True
        assert message == ""

    def test_nonexistent_file(self) -> None:
        """Test with a nonexistent file."""
        result, message = validate_requirements_file("/path/to/nonexistent/requirements.txt")
        assert result is False
        assert "not found" in message

    def test_file_size_check_failure_path(self, temp_requirements_file: Path, mocker: MockerFixture) -> None:
        """Test the failure path for file size check."""
        with open(temp_requirements_file, "w") as f:
            f.write("sample content")

        # Mock _check_file_size to return a failure
        mocker.patch(
            "app.services.tool.validator._check_file_size", return_value=(False, "File too large mock message")
        )

        result, message = validate_requirements_file(str(temp_requirements_file))
        assert result is False
        assert message == "File too large mock message"

    def test_requirements_count_check_failure_path(self, temp_requirements_file: Path, mocker: MockerFixture) -> None:
        """Test the failure path for requirements count check."""
        # Create a valid file
        with open(temp_requirements_file, "w") as f:
            f.write("sample=1.0.0\n")

        # Mock _check_requirements_count to return a failure
        mocker.patch(
            "app.services.tool.validator._check_requirements_count",
            return_value=(False, "Too many requirements mock message"),
        )

        result, message = validate_requirements_file(str(temp_requirements_file))
        assert result is False
        assert message == "Too many requirements mock message"

    def test_file_with_dangerous_pattern(self, temp_requirements_file: Path) -> None:
        """Test a file with dangerous patterns."""
        with open(temp_requirements_file, "w") as f:
            f.writelines(
                [
                    "normal-package==1.0.0\n",
                    "setuptools==1.0.0\n",  # Dangerous
                ]
            )

        result, message = validate_requirements_file(str(temp_requirements_file))
        assert result is False
        assert "unsafe package" in message.lower()

    def test_file_with_unsafe_pattern(self, temp_requirements_file: Path) -> None:
        """Test a file with unsafe patterns."""
        with open(temp_requirements_file, "w") as f:
            f.writelines(
                [
                    "normal-package==1.0.0\n",
                    "git+https://github.com/example/package.git\n",  # Unsafe
                ]
            )

        result, message = validate_requirements_file(str(temp_requirements_file))
        assert result is False
        assert "unsupported installation method" in message.lower()

    def test_exception_handling(self, temp_requirements_file: Path, mocker: MockerFixture) -> None:
        """Test exception handling."""

        # Mock open to raise an exception
        mocker.patch("builtins.open", side_effect=OSError("Simulated error"))

        result, message = validate_requirements_file(str(temp_requirements_file))
        assert result is False
        assert "error validating requirements" in message.lower()
