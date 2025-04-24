"""
Validation utilities for tool packages.
"""

from pathlib import Path

# Constants for validation
MAX_FILE_SIZE_MB = 150
MAX_REQUIREMENTS_COUNT = 100
DANGEROUS_PATTERNS: set[str] = {
    "setuptools==",
    "setuptools>=",
    "setuptools<=",  # Can be used for arbitrary code execution
    "pip==",
    "pip>=",
    "pip<=",  # Should not modify pip itself
    "wheel==",
    "wheel>=",
    "wheel<=",  # Potential for arbitrary code execution
    "./",  # Local directory installation - unsafe
}

UNSAFE_PATTERNS: set[str] = {
    "-e ",  # Editable installs
    "git+",  # Git repositories
    "http://",  # HTTP URLs
    "https://",  # HTTPS URLs
    "`",  # Command substitution
    "$",  # Variable interpolation
    "&",  # Command chaining
    "|",  # Command piping
    ";",  # Command sequencing
}


def _check_file_exists(requirements_file: Path) -> tuple[bool, str]:
    """Check if requirements file exists."""
    if not requirements_file.exists():
        return False, f"Requirements file not found: {requirements_file}"
    return True, ""


def _check_file_size(requirements_file: Path) -> tuple[bool, str]:
    """Check if requirements file size is within limits."""
    file_size_mb = requirements_file.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"Requirements file too large: {file_size_mb}MB (max: {MAX_FILE_SIZE_MB}MB)"
    return True, ""


def _check_requirements_count(requirements: list[str]) -> tuple[bool, str]:
    """Check if the number of requirements is within limits."""
    non_empty_lines = [line for line in requirements if line.strip() and not line.strip().startswith("#")]
    if len(non_empty_lines) > MAX_REQUIREMENTS_COUNT:
        return False, f"Too many packages in requirements file: {len(non_empty_lines)} (max: {MAX_REQUIREMENTS_COUNT})"
    return True, ""


def _check_for_dangerous_patterns(requirements: list[str]) -> tuple[bool, str]:
    """Check if requirements contain dangerous patterns."""
    for requirement in requirements:
        req = requirement.strip()
        if not req or req.startswith("#"):
            continue

        for pattern in DANGEROUS_PATTERNS:
            if pattern in req:
                return False, f"Potentially unsafe package or pattern detected: {req}"
    return True, ""


def _check_for_unsafe_patterns(requirements: list[str]) -> tuple[bool, str]:
    """Check if requirements contain unsafe installation methods."""
    for requirement in requirements:
        req = requirement.strip()
        if not req or req.startswith("#"):
            continue

        if any(p in req for p in UNSAFE_PATTERNS):
            return False, f"Unsupported installation method detected: {req}"
    return True, ""


def validate_requirements_file(requirements_path: str) -> tuple[bool, str]:
    """
    Validate requirements.txt file to ensure it doesn't contain unsafe packages.

    Args:
        requirements_path: Path to the requirements.txt file

    Returns:
        tuple of (is_valid, error_message)
    """
    requirements_file = Path(requirements_path)

    # File existence check
    is_valid, error_message = _check_file_exists(requirements_file)
    if not is_valid:
        return False, error_message

    # File size check
    is_valid, error_message = _check_file_size(requirements_file)
    if not is_valid:
        return False, error_message

    try:
        # Read requirements file
        with open(requirements_file) as file:
            requirements = file.readlines()

        # Number of requirements check
        is_valid, error_message = _check_requirements_count(requirements)
        if not is_valid:
            return False, error_message

        # Dangerous patterns check
        is_valid, error_message = _check_for_dangerous_patterns(requirements)
        if not is_valid:
            return False, error_message

        # Unsafe patterns check
        is_valid, error_message = _check_for_unsafe_patterns(requirements)
        if not is_valid:
            return False, error_message

        return True, ""

    except Exception as e:
        return False, f"Error validating requirements: {str(e)}"
