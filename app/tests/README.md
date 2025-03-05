# Test Utilities

This directory contains shared utilities for testing the application.

## AsyncMock

`AsyncMock` is a mock class designed for asynchronous functions. It allows you to easily mock async functions in your tests.

### Features

- Can be awaited like a real async function
- Tracks call count and arguments
- Supports various types of return values and side effects
- Provides assertion methods for verifying calls

### Usage Examples

#### Basic usage with a return value

```python
from app.tests.utils import AsyncMock

# Create a mock with a fixed return value
mock_function = AsyncMock(return_value={"status": "success"})

# Use in an async test
async def test_something():
    result = await mock_function("arg1", key="value")
    assert result == {"status": "success"}
    assert mock_function.call_count == 1
    mock_function.assert_called_with("arg1", key="value")
```

#### Mocking with exceptions

```python
# Create a mock that raises an exception
error_mock = AsyncMock(side_effect=ValueError("Test error"))

async def test_error_handling():
    with pytest.raises(ValueError):
        await error_mock()
```

#### Using multiple return values

```python
# Create a mock with a sequence of return values
sequence_mock = AsyncMock(side_effect=[
    {"first": True},
    {"second": True},
    ValueError("Third call raises")
])

async def test_sequence():
    # First call
    assert await sequence_mock() == {"first": True}
    
    # Second call
    assert await sequence_mock() == {"second": True}
    
    # Third call raises an exception
    with pytest.raises(ValueError):
        await sequence_mock()
```

#### Dynamic return values with a callable

```python
# Create a mock that uses the arguments to determine the return value
def calculate_result(*args, **kwargs):
    return sum(args)

dynamic_mock = AsyncMock(side_effect=calculate_result)

async def test_dynamic():
    assert await dynamic_mock(1, 2, 3) == 6
    assert await dynamic_mock(4, 5) == 9
```

### Integrating with pytest-mock

When using pytest-mock, you can patch async functions with AsyncMock:

```python
def test_with_mocker(mocker):
    # Patch an async function
    mocker.patch("my_module.async_function", new=AsyncMock(return_value="mocked"))
    
    # Test the code that uses the patched function
    # ...
```

## Contributing

When adding new test utilities:

1. Place the utility in `app/tests/utils.py`
2. Add comprehensive tests in `app/tests/test_utils.py`
3. Update this README with examples of how to use the new utility 