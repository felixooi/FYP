# Unit Testing Guide

## Overview
This project uses **pytest** for unit testing. The test suite covers all major modules with 45+ test cases.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_data_ingestion.py         # 8 tests
├── test_data_cleaning.py          # 12 tests
├── test_imbalance_handling.py     # 10 tests
├── test_data_transformation.py    # 8 tests
└── test_data_partition.py         # 7 tests
```

## Installation

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_data_cleaning.py
```

### Run specific test class
```bash
pytest tests/test_data_cleaning.py::TestHandleMissingValues
```

### Run with coverage report
```bash
pytest --cov=modules --cov-report=html
```

### Run verbose mode
```bash
pytest -v
```

## Test Coverage by Module

| Module | Functions | Tests | Coverage |
|--------|-----------|-------|----------|
| data_ingestion | 2 | 8 | Core functions |
| data_cleaning | 5 | 12 | All functions |
| imbalance_handling | 4 | 10 | All methods |
| data_transformation | 4 | 8 | Key functions |
| data_partition | 4 | 7 | Main pipeline |

## Key Test Fixtures (conftest.py)

- `sample_df`: Clean 100-row employee dataset
- `sample_df_with_missing`: Dataset with null values
- `sample_df_with_duplicates`: Dataset with duplicate rows
- `imbalanced_df`: Dataset with 90:10 class imbalance
- `temp_csv_path`: Temporary file path for I/O tests

## Why pytest?

✅ **Simple syntax** - No boilerplate code  
✅ **Powerful fixtures** - Reusable test data  
✅ **Parametrization** - Test multiple scenarios easily  
✅ **Great reporting** - Clear failure messages  
✅ **Industry standard** - Used by major Python projects  

## Example Test

```python
def test_handle_missing_values(sample_df_with_missing):
    result = handle_missing_values(sample_df_with_missing)
    assert result['Training_Hours'].isnull().sum() == 0
```

## Best Practices

1. **Isolate tests** - Each test should be independent
2. **Use fixtures** - Avoid code duplication
3. **Test edge cases** - Empty data, invalid inputs
4. **Mock external calls** - S3, API calls
5. **Keep tests fast** - Use small datasets
6. **Clear assertions** - One logical check per test

## CI/CD Integration

Add to your pipeline:
```yaml
- name: Run tests
  run: pytest --cov=modules --cov-report=xml
```
