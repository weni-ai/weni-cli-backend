"""
Tests for the utility functions and classes in app/tests/utils.py
"""

import asyncio
from typing import Any

import pytest

from app.tests.utils import AsyncMock


@pytest.mark.asyncio
async def test_async_mock_basic_return_value() -> None:
    """Test AsyncMock with a basic return value."""
    expected_value = {"result": "success"}
    mock = AsyncMock(return_value=expected_value)

    # Call the mock and check return value
    result = await mock("arg1", "arg2", kwarg1="value1")

    assert result == expected_value
    assert mock.call_count == 1
    assert mock.calls == 1  # For backwards compatibility
    assert mock.call_args == (("arg1", "arg2"), {"kwarg1": "value1"})
    assert len(mock.call_args_list) == 1
    assert mock.call_args_list[0] == (("arg1", "arg2"), {"kwarg1": "value1"})


@pytest.mark.asyncio
async def test_async_mock_exception_side_effect() -> None:
    """Test AsyncMock with an exception side effect."""
    error_message = "Test error"
    mock = AsyncMock(side_effect=ValueError(error_message))

    # Ensure the exception is raised when the mock is called
    with pytest.raises(ValueError) as exc_info:
        await mock()

    assert str(exc_info.value) == error_message
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_async_mock_list_side_effects() -> None:
    """Test AsyncMock with a list of side effects."""
    # Constants for expected call counts
    first_call = 1
    second_call = 2
    third_call = 3
    fourth_call = 4

    side_effects = [{"first": True}, {"second": True}, ValueError("Error after values")]
    mock = AsyncMock(side_effect=side_effects)

    # First call should return the first value
    assert await mock() == {"first": True}
    assert mock.call_count == first_call

    # Second call should return the second value
    assert await mock() == {"second": True}
    assert mock.call_count == second_call

    # Third call should raise the exception
    with pytest.raises(ValueError) as value_exc_info:
        await mock()
    assert str(value_exc_info.value) == "Error after values"
    assert mock.call_count == third_call

    # Fourth call should raise an IndexError since the list is exhausted
    with pytest.raises(IndexError) as index_exc_info:
        await mock()
    assert "AsyncMock side_effect list is empty" in str(index_exc_info.value)
    assert mock.call_count == fourth_call


@pytest.mark.asyncio
async def test_async_mock_callable_side_effect() -> None:
    """Test AsyncMock with a callable side effect."""

    def callable_side_effect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"args": args, "kwargs": kwargs}

    mock = AsyncMock(side_effect=callable_side_effect)

    # Call the mock and check that the side effect function is used
    result = await mock("arg1", kwarg1="value1")

    assert result == {"args": ("arg1",), "kwargs": {"kwarg1": "value1"}}
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_async_mock_async_callable_side_effect() -> None:
    """Test AsyncMock with an async callable side effect."""

    async def async_side_effect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.01)  # Small delay to simulate async work
        return {"async_args": args, "async_kwargs": kwargs}

    mock = AsyncMock(side_effect=async_side_effect)

    # Call the mock and check that the async side effect is awaited
    result = await mock("arg1", kwarg1="value1")

    assert result == {"async_args": ("arg1",), "async_kwargs": {"kwarg1": "value1"}}
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_async_mock_assertion_methods() -> None:
    """Test the assertion methods of AsyncMock."""
    mock = AsyncMock(return_value="result")

    # Call the mock once
    await mock("arg1", kwarg1="value1")

    # Test assert_called_once
    mock.assert_called_once()

    # Test assert_called_with
    mock.assert_called_with("arg1", kwarg1="value1")

    # Call it again
    await mock("arg2", kwarg2="value2")

    # assert_called_once should now fail
    with pytest.raises(AssertionError):
        mock.assert_called_once()

    # assert_called_with should check the latest call
    mock.assert_called_with("arg2", kwarg2="value2")

    # Checking with wrong arguments should fail
    with pytest.raises(AssertionError):
        mock.assert_called_with("wrong_arg")

    with pytest.raises(AssertionError):
        mock.assert_called_with("arg2", kwarg2="wrong_value")


@pytest.mark.asyncio
async def test_async_mock_call_args_when_not_called() -> None:
    """Test call_args property when the mock has not been called."""
    mock = AsyncMock()

    assert mock.call_args is None

    with pytest.raises(AssertionError):
        mock.assert_called_with("any_arg")
