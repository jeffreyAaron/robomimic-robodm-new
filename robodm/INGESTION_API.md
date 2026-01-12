# RoboDM Data Ingestion API

## Overview

The RoboDM Data Ingestion API provides a flexible, Ray-powered system for converting various data sources into VLA datasets with parallel processing support. This API addresses the challenge of transforming custom data formats (like Philips physiological data) into the robodm trajectory format while maintaining high performance through Ray-based parallelization.

## Key Benefits

- **Minimal Code Changes**: Convert existing PyTorch datasets, iterators, or custom data sources with 1-2 lines of code
- **Automatic Parallelization**: Ray-based parallel processing handles scaling automatically
- **Flexible Adapters**: Built-in adapters for common data source types
- **Custom Transformations**: Easy to define custom data transformation logic
- **Modular Design**: Clean separation between data ingestion and the core robodm library

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│   Ingestion     │───▶│  VLA Dataset    │
│                 │    │   Interface     │    │                 │
│ • PyTorch       │    │                 │    │ • Ray-powered   │
│ • Iterators     │    │ • Transform     │    │ • Trajectory    │
│ • Files         │    │ • Validate      │    │   format        │
│ • Custom        │    │ • Group         │    │ • Parallel      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Ray Workers    │
                       │                 │
                       │ • Parallel      │
                       │   Processing    │
                       │ • Trajectory    │
                       │   Creation      │
                       └─────────────────┘
```

## Quick Start

### 1. PyTorch Dataset (Simplest)

```python
from robodm.ingestion import create_vla_dataset_from_source

# Your existing PyTorch dataset
pytorch_dataset = MyExistingDataset()

# Convert to VLA dataset with one line!
vla_dataset = create_vla_dataset_from_source(
    data_source=pytorch_dataset,
    output_directory="./my_trajectories",
    num_workers=4
)

# Use immediately with existing VLA API
for batch in vla_dataset.iter_batches(batch_size=32):
    # Your training loop here
    pass
```

### 2. Custom Data with Transform Function

```python
def transform_my_data(item):
    """Transform your data format to robodm format."""
    return {
        "sensor_data": item.sensor_readings,
        "image": item.camera_frame,
        "metadata": {"timestamp": item.timestamp}
    }

# Any data source + transform function
vla_dataset = create_vla_dataset_from_source(
    data_source=my_data_list,  # List, iterator, etc.
    transform_fn=transform_my_data,
    output_directory="./my_trajectories",
    num_workers=8
)
```

### 3. Custom Ingester (Full Control)

```python
from robodm.ingestion import DataIngestionInterface

class MyCustomIngester(DataIngestionInterface):
    def get_data_items(self):
        # Return list of items to process
        return [...]
    
    def transform_item(self, item):
        # Transform item to robodm format
        return {"feature1": ..., "feature2": ...}

# Use your custom ingester
ingester = MyCustomIngester()
vla_dataset = create_vla_dataset_from_source(ingester)
```

## Core Interfaces

### DataIngestionInterface

The main interface users implement to define their data transformation logic:

```python
from abc import ABC, abstractmethod

class DataIngestionInterface(ABC):
    @abstractmethod
    def get_data_items(self) -> List[Any]:
        """Return list of data items to process."""
        pass
    
    @abstractmethod
    def transform_item(self, item: Any) -> Dict[str, Any]:
        """Transform item into robodm trajectory format."""
        pass
    
    # Optional methods for customization
    def group_items_into_trajectories(self, items):
        """Group items into trajectory files."""
        return [[item] for item in items]  # Default: one item per trajectory
    
    def get_trajectory_filename(self, trajectory_group, index):
        """Generate trajectory filename."""
        return f"trajectory_{index:06d}"
    
    def validate_transformed_data(self, data):
        """Validate transformed data before adding to trajectory."""
        return True
```

### IngestionConfig

Configuration for the ingestion process:

```python
@dataclass
class IngestionConfig:
    # Output configuration
    output_directory: str
    trajectory_prefix: str = "trajectory"
    
    # Parallel processing
    num_workers: int = 4
    batch_size: int = 1
    
    # Trajectory configuration
    time_unit: str = "ms"
    video_codec: str = "auto"
    raw_codec: Optional[str] = None
    
    # Data processing
    shuffle_items: bool = False
    max_items_per_trajectory: Optional[int] = None
```

## Built-in Adapters

### PyTorchDatasetAdapter

For PyTorch `Dataset` objects:

```python
from robodm.ingestion import PyTorchDatasetAdapter

adapter = PyTorchDatasetAdapter(
    dataset=pytorch_dataset,
    transform_fn=my_transform_function,  # Optional
    group_size=100,  # Items per trajectory
)
```

### IteratorAdapter

For iterators and generators:

```python
from robodm.ingestion import IteratorAdapter

def my_data_generator():
    for i in range(10000):
        yield generate_data_item(i)

adapter = IteratorAdapter(
    iterator_factory=my_data_generator,
    transform_fn=transform_function,
    max_items=1000,  # Optional limit
)
```

### FileListAdapter

For processing files:

```python
from robodm.ingestion import FileListAdapter

file_paths = ["data1.json", "data2.json", ...]

adapter = FileListAdapter(
    file_paths=file_paths,
    transform_fn=load_and_transform_file,
    group_size=50,  # Files per trajectory
)
```

### CallableAdapter

For callable functions that generate data:

```python
from robodm.ingestion import CallableAdapter

def generate_data():
    return [create_item(i) for i in range(1000)]

adapter = CallableAdapter(
    data_generator=generate_data,
    transform_fn=process_item,
)
```

## Factory Functions

### Main Factory Function

```python
create_vla_dataset_from_source(
    data_source,              # Any supported data source
    output_directory=None,    # Where to save trajectories
    transform_fn=None,        # Optional transformation
    group_size=1,            # Items per trajectory
    num_workers=4,           # Parallel workers
    return_vla_dataset=True, # Return VLADataset vs file paths
    **kwargs                 # Additional config
)
```

### Specialized Factory Functions

```python
# PyTorch datasets
create_vla_dataset_from_pytorch_dataset(
    dataset, trajectories_per_dataset=1, **kwargs
)

# File lists
create_vla_dataset_from_file_list(
    file_paths, transform_fn, files_per_trajectory=100, **kwargs
)

# Iterators
create_vla_dataset_from_iterator(
    iterator_factory, max_items=None, items_per_trajectory=100, **kwargs
)

# Callables
create_vla_dataset_from_callable(
    data_generator, items_per_trajectory=100, **kwargs
)
```

## Ray Integration

The system leverages Ray for:

- **Parallel Data Processing**: Multiple workers process trajectory groups simultaneously
- **Automatic Scaling**: Ray handles worker management and task distribution
- **Memory Management**: Efficient handling of large datasets
- **Fault Tolerance**: Built-in error handling and recovery

### Ray Configuration

```python
# Custom Ray initialization
ray_config = {
    "num_cpus": 16,
    "object_store_memory": 4_000_000_000,  # 4GB
}

vla_dataset = create_vla_dataset_from_source(
    data_source=my_dataset,
    ray_init_kwargs=ray_config,
    num_workers=8,
)
```

## Use Cases

### 1. Physiological Data (like Philips)

```python
class PhilipsIngester(DataIngestionInterface):
    def __init__(self, data_directory, sensor_filter):
        self.data_directory = data_directory
        self.sensor_filter = sensor_filter
    
    def get_data_items(self):
        # Discover all data files/segments
        return self._scan_philips_data()
    
    def transform_item(self, segment_info):
        # Load and transform physiological signals
        return {
            "ecg_lead_ii": self._load_signal(segment_info, "II"),
            "ecg_lead_avl": self._load_signal(segment_info, "aVL"),
            "visualization": self._create_plot(segment_info),
        }

ingester = PhilipsIngester("/data/philips", ["II", "aVL", "V"])
vla_dataset = create_vla_dataset_from_source(ingester)
```

### 2. Computer Vision

```python
# Existing PyTorch vision dataset
vision_dataset = torchvision.datasets.CIFAR10(...)

def vision_transform(data_tuple):
    image, label = data_tuple
    return {
        "image": image.numpy().transpose(1, 2, 0),  # CHW -> HWC
        "label": label,
        "augmented_image": apply_augmentation(image),
    }

vla_dataset = create_vla_dataset_from_source(
    vision_dataset,
    transform_fn=vision_transform,
    group_size=1000,  # 1000 images per trajectory
)
```

### 3. Time Series

```python
def load_timeseries_files():
    """Load time series data from files."""
    for filepath in glob.glob("timeseries/*.csv"):
        df = pd.read_csv(filepath)
        for i in range(0, len(df), 100):  # 100-sample windows
            yield {
                "sequence": df.iloc[i:i+100].values,
                "metadata": {"file": filepath, "window": i//100}
            }

vla_dataset = create_vla_dataset_from_source(
    load_timeseries_files,
    group_size=50,  # 50 windows per trajectory
)
```

### 4. Robotics Data

```python
class RobotDataIngester(DataIngestionInterface):
    def transform_item(self, episode_path):
        episode_data = load_episode(episode_path)
        return {
            "observation": episode_data.observations,
            "action": episode_data.actions,
            "reward": episode_data.rewards,
            "camera_rgb": episode_data.camera_frames,
            "gripper_pos": episode_data.gripper_positions,
        }

robot_ingester = RobotDataIngester()
vla_dataset = create_vla_dataset_from_source(robot_ingester)
```

## Performance Optimization

### Memory Management

```python
# For large datasets
config = IngestionConfig(
    output_directory="./large_dataset",
    num_workers=16,
    raw_codec="rawvideo_pyarrow",  # Efficient compression
    max_items_per_trajectory=10000,  # Larger trajectories
)
```

### Parallel Processing

```python
# Optimize for your hardware
optimal_workers = min(os.cpu_count(), 16)  # Don't exceed CPU count

vla_dataset = create_vla_dataset_from_source(
    data_source=large_dataset,
    num_workers=optimal_workers,
    group_size=1000,  # Balance between memory and I/O
)
```

### Streaming for Very Large Datasets

```python
def streaming_data_generator():
    """Generator for datasets too large for memory."""
    for chunk in load_data_in_chunks():
        for item in chunk:
            yield item

vla_dataset = create_vla_dataset_from_source(
    streaming_data_generator,
    max_items=1_000_000,  # Process subset
    num_workers=8,
)
```

## Integration with Existing VLA API

The ingestion API produces standard VLA datasets that work with all existing robodm functionality:

```python
# Create VLA dataset with ingestion API
vla_dataset = create_vla_dataset_from_source(my_data_source)

# Use with existing VLA functionality
train_dataset, val_dataset = vla_dataset.split(0.8, 0.2)

# Iterate normally
for batch in train_dataset.iter_batches(batch_size=32):
    # Training loop
    pass

# Load data
data = val_dataset.load(desired_frequency=10.0)

# Get statistics
stats = vla_dataset.get_stats()
```

## Migration Guide

### From Existing PyTorch Code

```python
# Before (PyTorch)
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # Training loop
    pass

# After (RoboDM with minimal changes)
dataset = MyDataset()
vla_dataset = create_vla_dataset_from_source(dataset)

for batch in vla_dataset.iter_batches(batch_size=32):
    # Same training loop!
    pass
```

### From Custom Data Loaders

```python
# Before (Custom loader)
class MyDataLoader:
    def __iter__(self):
        for item in self.load_data():
            yield self.process_item(item)

# After (RoboDM ingestion)
def my_data_generator():
    loader = MyDataLoader()
    return list(loader)

vla_dataset = create_vla_dataset_from_source(
    my_data_generator,
    transform_fn=lambda item: {"data": item}
)
```

## Error Handling

The ingestion system provides robust error handling:

```python
class MyIngester(DataIngestionInterface):
    def validate_transformed_data(self, data):
        """Custom validation logic."""
        required_keys = ["sensor1", "sensor2"]
        if not all(key in data for key in required_keys):
            return False
        return True
    
    def transform_item(self, item):
        try:
            return self._transform_logic(item)
        except Exception as e:
            logger.warning(f"Failed to transform {item}: {e}")
            return {}  # Return empty dict to skip
```

## Best Practices

1. **Start Simple**: Use `create_vla_dataset_from_source()` with automatic detection first
2. **Custom Transforms**: Define clear transformation functions for your data format
3. **Grouping Strategy**: Choose group sizes that balance memory usage and I/O efficiency
4. **Validation**: Implement data validation to catch issues early
5. **Monitoring**: Use logging to track ingestion progress and identify bottlenecks
6. **Testing**: Test with small datasets first before scaling up

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `group_size` or increase `num_workers`
2. **Slow Processing**: Check if transformation functions are efficient
3. **Ray Errors**: Ensure Ray is properly installed and initialized
4. **File Permissions**: Check write permissions for output directory

### Performance Tuning

```python
# Profile your transformation function
import time

def timed_transform(item):
    start = time.time()
    result = my_transform(item)
    print(f"Transform took {time.time() - start:.3f}s")
    return result

vla_dataset = create_vla_dataset_from_source(
    data_source=my_data,
    transform_fn=timed_transform,
)
```

## Future Extensions

The ingestion API is designed to be extensible:

- **New Adapters**: Easy to add adapters for new data source types
- **Custom Backends**: Support for different storage backends
- **Streaming Support**: Enhanced streaming for infinite datasets
- **Cloud Integration**: Built-in support for cloud storage and processing

This architecture provides a clean separation between domain-specific data loaders (like your Philips loader) and the core robodm library, while enabling powerful parallel processing through Ray. 