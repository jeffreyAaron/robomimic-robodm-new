"""Pytest configuration and fixture registration."""

import sys

import pytest

# Import all fixtures from test_fixtures
from .test_fixtures import (benchmark_dataset, large_sample_data,
                            mock_filesystem, mock_time_provider,
                            sample_dict_of_lists, sample_trajectory_data,
                            temp_dir)

# Re-export fixtures so pytest can find them
__all__ = [
    "mock_filesystem",
    "mock_time_provider",
    "temp_dir",
    "sample_trajectory_data",
    "sample_dict_of_lists",
    "large_sample_data",
    "benchmark_dataset",
]


# each test runs on cwd to its temp dir
@pytest.fixture(autouse=True)
def go_to_tmpdir(request):
    # Get the fixture dynamically by its name.
    tmpdir = request.getfixturevalue("tmpdir")
    # ensure local test created packages can be imported
    sys.path.insert(0, str(tmpdir))
    # Chdir only for the duration of the test.
    with tmpdir.as_cwd():
        yield
