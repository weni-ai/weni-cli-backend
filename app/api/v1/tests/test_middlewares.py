"""
Tests for API middlewares.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response, status

from app.api.v1.middlewares import VersionCheckMiddleware


class TestVersionCheckMiddleware:
    """Tests for the VersionCheckMiddleware class."""

    @pytest.fixture
    def middleware(self) -> VersionCheckMiddleware:
        """Create a middleware instance for testing."""
        return VersionCheckMiddleware()

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create a mock request for testing."""
        mock = MagicMock(spec=Request)
        mock.url = MagicMock()
        mock.url.path = "/api/v1/some/endpoint"
        mock.headers = {}
        return mock

    @pytest.fixture
    def mock_call_next(self) -> AsyncMock:
        """Create a mock call_next function for testing."""
        mock_response = Response(status_code=status.HTTP_200_OK)
        return AsyncMock(return_value=mock_response)

    @pytest.mark.asyncio
    async def test_missing_cli_version_header(
        self, middleware: VersionCheckMiddleware, mock_request: MagicMock, mock_call_next: AsyncMock
    ) -> None:
        """Test middleware with missing X-CLI-Version header."""
        mock_request.headers = {}

        response = await middleware(mock_request, mock_call_next)

        assert response.status_code == status.HTTP_426_UPGRADE_REQUIRED
        assert response.body == b"Upgrade required, please update your CLI to the latest version"
        mock_call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_cli_version_format(
        self, middleware: VersionCheckMiddleware, mock_request: MagicMock, mock_call_next: AsyncMock
    ) -> None:
        """Test middleware with invalid X-CLI-Version format."""
        mock_request.headers = {"X-CLI-Version": "not.a.valid.version"}

        response = await middleware(mock_request, mock_call_next)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.body == b"Invalid version format: not.a.valid.version"
        mock_call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_older_cli_version(
        self, middleware: VersionCheckMiddleware, mock_request: MagicMock, mock_call_next: AsyncMock
    ) -> None:
        """Test middleware with older X-CLI-Version than minimum required."""
        with patch("app.core.config.settings.CLI_MINIMUM_VERSION", "1.0.0"):
            mock_request.headers = {"X-CLI-Version": "0.9.0"}

            response = await middleware(mock_request, mock_call_next)

            assert response.status_code == status.HTTP_426_UPGRADE_REQUIRED
            assert response.body == b"Upgrade required. Minimum version: 1.0.0, your version: 0.9.0"
            mock_call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_equal_cli_version(
        self, middleware: VersionCheckMiddleware, mock_request: MagicMock, mock_call_next: AsyncMock
    ) -> None:
        """Test middleware with equal X-CLI-Version as minimum required."""
        with patch("app.core.config.settings.CLI_MINIMUM_VERSION", "1.0.0"):
            mock_request.headers = {"X-CLI-Version": "1.0.0"}

            await middleware(mock_request, mock_call_next)

            mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_newer_cli_version(
        self, middleware: VersionCheckMiddleware, mock_request: MagicMock, mock_call_next: AsyncMock
    ) -> None:
        """Test middleware with newer X-CLI-Version than minimum required."""
        with patch("app.core.config.settings.CLI_MINIMUM_VERSION", "1.0.0"):
            mock_request.headers = {"X-CLI-Version": "1.1.0"}

            await middleware(mock_request, mock_call_next)

            mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_health_endpoint_bypass(
        self, middleware: VersionCheckMiddleware, mock_request: MagicMock, mock_call_next: AsyncMock
    ) -> None:
        """Test middleware bypasses version check for health endpoints."""
        mock_request.url.path = "/api/v1/health"
        mock_request.headers = {}  # No version header

        await middleware(mock_request, mock_call_next)

        mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.parametrize(
        "client_version,min_version,expected",
        [
            ("1.0.0", "1.0.0", True),  # Equal versions
            ("1.1.0", "1.0.0", True),  # Higher major version
            ("1.0.1", "1.0.0", True),  # Higher minor version
            ("0.9.0", "1.0.0", False),  # Lower major version
            ("1.0.0", "1.1.0", False),  # Lower minor version
            ("2.0.0", "1.9.9", True),  # Higher major overrides minor
            ("1.0", "1.0.0", True),  # Missing version component
            ("1", "1.0.0", True),  # Single version component
            # PEP 0440 specific version formats
            ("1.0.0.post1", "1.0.0", True),  # Post-release
            ("1.0.0.dev1", "1.0.0", False),  # Dev release
            ("1.0.0a1", "1.0.0", False),  # Alpha release
            ("1.0.0b1", "1.0.0", False),  # Beta release
            ("1.0.0rc1", "1.0.0", False),  # Release candidate
            ("1.0.0.post1", "1.0.0.post2", False),  # Post-release comparison
        ],
    )
    def test_version_compatibility(
        self, middleware: VersionCheckMiddleware, client_version: str, min_version: str, expected: bool
    ) -> None:
        """Test version compatibility logic with various version combinations."""
        assert middleware._is_version_compatible(client_version, min_version) == expected

    def test_invalid_version_format(self, middleware: VersionCheckMiddleware) -> None:
        """Test that invalid version formats raise ValueError."""
        with pytest.raises(ValueError):
            middleware._is_version_compatible("invalid", "1.0.0")

        with pytest.raises(ValueError):
            middleware._is_version_compatible("1.0.0", "invalid")
