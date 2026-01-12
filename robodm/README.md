# 🦊 Robo-DM

**An Efficient and Scalable Data Collection and Management Framework For Robotics Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/BerkeleyAutomation/robodm)](LICENSE)
[![Tests](https://github.com/BerkeleyAutomation/robodm/workflows/Tests/badge.svg)](https://github.com/BerkeleyAutomation/robodm/actions)

robodm is a high-performance robotics data management framework that enables efficient collection, storage, and retrieval of multimodal robotics trajectories. Built with speed 🚀 and memory efficiency 📈 in mind, robodm provides native support for various robotics data formats and cloud storage systems.

## ✨ Key Features

- **🚀 High Performance**: Optimized for speed with active metadata and lazily-loaded trajectory data
- **📈 Memory Efficient**: Smart data loading and compression strategies minimize memory usage
- **🎥 Advanced Video Compression**: Support for multiple codecs (H.264, H.265, AV1, FFV1) with automatic codec selection
- **🔄 Format Compatibility**: Native support for Open-X-Embodiment, HuggingFace datasets, RLDS, and HDF5
- **🎯 Flexible Data Types**: Handle images, videos, sensor data, and custom features seamlessly
- **🏗️ Distributed Ready**: Flexible dataset partitioning for distributed training workflows

## 🛠️ Installation

### Basic Installation

```bash
git clone https://github.com/BerkeleyAutomation/robodm.git
cd robodm
pip install -e .
```

### Installation with Optional Dependencies

```bash
# For HuggingFace integration
pip install -e .[hf]

# For Open-X-Embodiment support
pip install -e .[rtx]

# For AWS cloud storage
pip install -e .[aws]

# For PyTorch integration
pip install -e .[torch]

# Install all optional dependencies
pip install -e .[all]
```

## 🚀 Quick Start

### Basic Data Collection and Loading

```python
import numpy as np
import robodm

# Create a new trajectory for data collection
trajectory = robodm.Trajectory(path="/tmp/robot_demo.vla", mode="w")

# Collect multimodal robotics data
for step in range(100):
    # Add camera observations
    trajectory.add("camera/rgb", np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    trajectory.add("camera/depth", np.random.rand(480, 640).astype(np.float32))
    
    # Add robot state
    trajectory.add("robot/joint_positions", np.random.rand(7).astype(np.float32))
    trajectory.add("robot/joint_velocities", np.random.rand(7).astype(np.float32))
    trajectory.add("robot/end_effector_pose", np.random.rand(4, 4).astype(np.float32))
    
    # Add action data
    trajectory.add("action/gripper_action", np.random.rand(1).astype(np.float32))

# Save and close the trajectory
trajectory.close()

# Load the trajectory for training
trajectory = robodm.Trajectory(path="/tmp/robot_demo.vla", mode="r")
data = trajectory.load()

print(f"Loaded trajectory with {len(data['camera/rgb'])} timesteps")
print(f"Camera RGB shape: {data['camera/rgb'][0].shape}")
print(f"Joint positions shape: {data['robot/joint_positions'][0].shape}")
```

### Batch Data Creation

```python
import robodm

# Create trajectory from dictionary of lists
data = {
    "observation/image": [np.random.randint(0, 255, (224, 224, 3)) for _ in range(50)],
    "observation/state": [np.random.rand(10) for _ in range(50)],
    "action": [np.random.rand(7) for _ in range(50)],
}

trajectory = robodm.Trajectory.from_dict_of_lists(
    data=data,
    path="/tmp/batch_trajectory.vla",
    video_codec="libaom-av1"  # Use AV1 codec for efficient compression
)
```

### Advanced Configuration

```python
import robodm

# Configure video compression settings
trajectory = robodm.Trajectory(
    path="/tmp/compressed_demo.vla",
    mode="w",
    video_codec="libx265",  # Use H.265 codec
    codec_options={
        "crf": "23",        # Quality setting (lower = higher quality)
        "preset": "fast"    # Encoding speed
    }
)

# Use hierarchical feature names
trajectory.add("sensors/lidar/points", lidar_data)
trajectory.add("sensors/camera/front/rgb", front_camera)
trajectory.add("sensors/camera/wrist/rgb", wrist_camera)
trajectory.add("control/arm/joint_positions", joint_positions)
```

## 🎥 Video Codec Support

robodm supports multiple video codecs for efficient storage of visual data:

| Codec | Use Case | Compression | Quality |
|-------|----------|-------------|---------|
| `rawvideo` | Lossless, fast I/O | None | Perfect |
| `ffv1` | Lossless compression | High | Perfect |
| `libx264` | General purpose | Very High | Excellent |
| `libx265` | Better compression | Very High | Excellent |
| `libaom-av1` | Best compression | Highest | Excellent |
| `auto` | Automatic selection | Optimal | Optimal |

```python
# Automatic codec selection based on data characteristics
trajectory = robodm.Trajectory(path="auto.vla", mode="w", video_codec="auto")

# Manual codec selection for specific needs
trajectory = robodm.Trajectory(path="lossless.vla", mode="w", video_codec="ffv1")
```

## 🧪 Development & Testing

### Running Tests

```bash
# Install development dependencies
pip install -e .[test]

# Run all tests
make test

# Run specific test categories
pytest tests/test_trajectory.py -v
pytest tests/test_loaders.py -v
```


## 📝 Examples

Explore the `examples/` directory for more detailed usage patterns:

- **[Basic Data Collection](./examples/data_collection_and_load.py)**: Simple data collection and loading
- **[Benchmark Scripts](./tests/)**: Performance testing and optimization

We are actively and heavily refactoring the code to make it more robust and maintainable. See commit `5bbb8b` for the prior ICRA submission. 



## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Setting up development environment
- Running tests and benchmarks  
- Code style and formatting
- Submitting pull requests

## 📄 License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.


## 📚 Citation

If you use robodm in your research, please cite:

```bibtex
@article{chen2025robo,
  title={Robo-DM: Data Management For Large Robot Datasets},
  author={Chen, Kaiyuan and Fu, Letian and Huang, David and Zhang, Yanxiang and Chen, Lawrence Yunliang and Huang, Huang and Hari, Kush and Balakrishna, Ashwin and Xiao, Ted and Sanketi, Pannag R and others},
  journal={arXiv preprint arXiv:2505.15558},
  year={2025}
}
```
