"""Test fixtures and mock implementations for robodm testing."""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from robodm import Trajectory
from robodm.trajectory_base import FileSystemInterface, TimeProvider


class MockFileSystem(FileSystemInterface):
    """Mock file system for testing."""

    def __init__(self):
        self.files = {}
        self.directories = set()

    def exists(self, path: str) -> bool:
        return path in self.files or path in self.directories

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        if not exist_ok and path in self.directories:
            raise FileExistsError(f"Directory {path} already exists")
        self.directories.add(path)

    def remove(self, path: str) -> None:
        if path in self.files:
            del self.files[path]
        else:
            raise FileNotFoundError(f"File {path} not found")

    def rename(self, src: str, dst: str) -> None:
        if src not in self.files:
            raise FileNotFoundError(f"File {src} not found")
        self.files[dst] = self.files[src]
        del self.files[src]

    def add_file(self, path: str, content: Any = None):
        """Add a file to the mock filesystem."""
        self.files[path] = content


class MockTimeProvider(TimeProvider):
    """Mock time provider for deterministic testing."""

    def __init__(self, initial_time: float = 0.0):
        self._current_time = initial_time
        self._time_calls = []

    def time(self) -> float:
        self._time_calls.append(self._current_time)
        return self._current_time

    def advance_time(self, seconds: float):
        """Advance the mock time."""
        self._current_time += seconds

    def set_time(self, time: float):
        """Set the current time."""
        self._current_time = time

    @property
    def call_count(self) -> int:
        return len(self._time_calls)


@pytest.fixture
def mock_filesystem():
    """Pytest fixture for mock filesystem."""
    return MockFileSystem()


@pytest.fixture
def mock_time_provider():
    """Pytest fixture for mock time provider."""
    return MockTimeProvider()


@pytest.fixture
def temp_dir():
    """Pytest fixture for temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_trajectory_data():
    """Sample trajectory data for testing."""
    return [
        {
            "observation": {
                "image": np.random.randint(0,
                                           255, (480, 640, 3),
                                           dtype=np.uint8),
                "joint_positions": np.random.random(7).astype(np.float32),
            },
            "action": np.random.random(7).astype(np.float32),
            "reward": np.float32(1.0),
        },
        {
            "observation": {
                "image": np.random.randint(0,
                                           255, (480, 640, 3),
                                           dtype=np.uint8),
                "joint_positions": np.random.random(7).astype(np.float32),
            },
            "action": np.random.random(7).astype(np.float32),
            "reward": np.float32(0.5),
        },
    ]


@pytest.fixture
def sample_dict_of_lists():
    """Sample dictionary of lists for testing."""
    return {
        "observation": {
            "image": [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            ],
            "joint_positions": [
                np.random.random(7).astype(np.float32),
                np.random.random(7).astype(np.float32),
            ],
        },
        "action": [
            np.random.random(7).astype(np.float32),
            np.random.random(7).astype(np.float32),
        ],
        "reward": [np.float32(1.0), np.float32(0.5)],
    }


@pytest.fixture
def large_sample_data():
    """Large sample data for benchmarking."""
    num_samples = 100
    return {
        "observation/image": [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(num_samples)
        ],
        "observation/joint_positions":
        [np.random.random(7).astype(np.float32) for _ in range(num_samples)],
        "action":
        [np.random.random(7).astype(np.float32) for _ in range(num_samples)],
        "reward": [np.float32(np.random.random()) for _ in range(num_samples)],
    }


class BenchmarkDataset:
    """Helper class for creating benchmark datasets."""

    @staticmethod
    def create_vla_dataset(path: str,
                           data: Dict[str, List[Any]],
                           video_codec: str = "auto"):
        """Create a VLA dataset file for testing."""
        traj = Trajectory.from_dict_of_lists(data,
                                             path,
                                             video_codec=video_codec)
        return traj

    @staticmethod
    def create_hdf5_dataset(path: str, data: Dict[str, List[Any]]):
        """Create an HDF5 dataset file."""
        import h5py

        with h5py.File(path, "w") as f:
            for key, values in data.items():
                if isinstance(values[0], np.ndarray):
                    stacked_data = np.stack(values)
                else:
                    stacked_data = np.array(values)
                f.create_dataset(key,
                                 data=stacked_data,
                                 compression="gzip",
                                 compression_opts=9)

    @staticmethod
    def get_file_size(path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(path)


@pytest.fixture
def benchmark_dataset():
    """Pytest fixture for benchmark dataset helper."""
    return BenchmarkDataset()
