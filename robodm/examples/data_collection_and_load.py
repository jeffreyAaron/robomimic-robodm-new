import os
import tempfile
import time

import numpy as np

import robodm

if __name__ == "__main__":
    path = os.path.join(tempfile.gettempdir(), "test_trajectory.vla")

    # Create a trajectory
    traj = robodm.Trajectory(path=path, mode="w")

    # Add some data
    for i in range(10):
        traj.add(
            "observation/image",
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        )
        traj.add("observation/state", np.random.rand(10).astype(np.float32))
        traj.add("action", np.random.rand(7).astype(np.float32))
        time.sleep(0.1)

    # Close the trajectory
    traj.close()

    print(f"Trajectory saved to {path}")

    # Load the trajectory
    traj = robodm.Trajectory(path=path, mode="r")
    data = traj.load()

    print(f"Loaded trajectory with {len(data['observation/image'])} timesteps")
    print(f"Image shape: {data['observation/image'][0].shape}")
    print(f"State shape: {data['observation/state'][0].shape}")
    print(f"Action shape: {data['action'][0].shape}")

    # Clean up
    os.remove(path)
