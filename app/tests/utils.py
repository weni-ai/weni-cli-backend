"""
Common test utilities for all tests in the application.
"""

from typing import Any


class AsyncMock:
    """
    A flexible mock for asynchronous functions.

    This mock can be awaited and supports various forms of return values and side effects.
    It also tracks call count and arguments for verification in tests.

    Examples:
        # Simple return value
        mock = AsyncMock(return_value={"success": True})

        # With a side effect exception
        mock = AsyncMock(side_effect=ValueError("Test error"))

        # With multiple return values
        mock = AsyncMock(side_effect=[{"first": True}, {"second": True}])

        # With a callable side effect
        mock = AsyncMock(side_effect=lambda x: {"input": x})
    """

    def __init__(self, return_value: Any = None, side_effect: Any = None) -> None:
        """
        Initialize the async mock.

        Args:
            return_value: Value to return when the mock is awaited
            side_effect: Side effect to apply when the mock is awaited.
                         Can be an exception, a callable, or a list of values.
        """
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_args_list: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.call_count = 0

        # For backwards compatibility with existing tests using .calls
        self.calls = 0

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Implement the callable interface to allow the mock to be awaited.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The configured return value or result of the side effect

        Raises:
            Exception: If the side effect is an exception
        """
        # Record the call
        self.call_args_list.append((args, kwargs))
        self.call_count += 1
        self.calls = self.call_count  # Keep both counters in sync

        # Handle side effects
        if self.side_effect is not None:
            # If side_effect is a list, pop the next value
            if isinstance(self.side_effect, list):
                if not self.side_effect:
                    raise IndexError("AsyncMock side_effect list is empty")
                result = self.side_effect.pop(0)
                if isinstance(result, Exception):
                    raise result
                return result

            # If side_effect is callable, call it with the arguments
            if callable(self.side_effect):
                try:
                    # Try to await it in case it's an async function
                    return await self.side_effect(*args, **kwargs)
                except TypeError:
                    # If it can't be awaited, call it directly
                    return self.side_effect(*args, **kwargs)

            # If side_effect is an exception, raise it
            if isinstance(self.side_effect, Exception):
                raise self.side_effect

            # Otherwise, just return the side_effect value
            return self.side_effect

        # Default to returning the configured return_value
        return self.return_value

    @property
    def call_args(self) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        """
        Get the arguments from the most recent call, or None if not called.

        Returns:
            A tuple of (args, kwargs) from the last call, or None if not called
        """
        if not self.call_args_list:
            return None
        return self.call_args_list[-1]

    def assert_called_once(self) -> None:
        """
        Assert that the mock was called exactly once.

        Raises:
            AssertionError: If the mock was not called exactly once
        """
        assert self.call_count == 1, f"Expected 1 call, got {self.call_count}"

    def assert_called_with(self, *args: Any, **kwargs: Any) -> None:
        """
        Assert that the most recent call was with the specified arguments.

        Args:
            *args: Expected positional arguments
            **kwargs: Expected keyword arguments

        Raises:
            AssertionError: If the mock was not called with the expected arguments
        """
        if not self.call_args:
            raise AssertionError("Mock was not called")

        call_args, call_kwargs = self.call_args
        assert args == call_args, f"Expected args: {args}, got: {call_args}"
        for key, value in kwargs.items():
            assert key in call_kwargs, f"Expected kwarg '{key}' not found"
            assert call_kwargs[key] == value, f"For kwarg '{key}', expected: {value}, got: {call_kwargs[key]}"
