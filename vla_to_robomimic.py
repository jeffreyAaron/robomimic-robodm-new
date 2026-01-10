import os
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm
import robodm

def convert_vla_to_hdf5(vla_dir, output_path):
    vla_files = sorted([f for f in os.listdir(vla_dir) if f.endswith(".vla")])
    if not vla_files:
        print(f"No .vla files found in {vla_dir}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")

    total_samples = 0
    env_meta = None

    print(f"Converting {len(vla_files)} VLA files from {vla_dir} to {output_path}...")

    for vla_file in tqdm(vla_files):
        traj_path = os.path.join(vla_dir, vla_file)
        
        # Load trajectory
        traj = robodm.Trajectory(traj_path, mode="r")
        data = traj.load()
        traj.close()

        # Use filename as demo key (e.g., demo_0.vla -> demo_0)
        demo_key = os.path.splitext(vla_file)[0]
        demo_grp = data_grp.create_group(demo_key)

        # Extract and preserve global metadata
        if "metadata/env_meta" in data:
            meta_str = data["metadata/env_meta"][0]
            if env_meta is None:
                env_meta = meta_str
                # In robomimic, env_args is a JSON string attribute on the 'data' group
                data_grp.attrs["env_args"] = env_meta
            elif env_meta != meta_str:
                # This could happen if VLA files from different environments are mixed
                pass

        # Extract and preserve per-trajectory metadata
        if "metadata/model_file" in data:
            demo_grp.attrs["model_file"] = data["metadata/model_file"][0]

        # Observations and other features
        obs_grp = demo_grp.create_group("obs")
        next_obs_grp = None
        
        num_samples = 0
        for key, value in data.items():
            if key.startswith("observation/"):
                obs_key = key.split("/", 1)[1]
                obs_grp.create_dataset(obs_key, data=value)
                num_samples = len(value)
            elif key.startswith("next_observation/"):
                if next_obs_grp is None:
                    next_obs_grp = demo_grp.create_group("next_obs")
                obs_key = key.split("/", 1)[1]
                next_obs_grp.create_dataset(obs_key, data=value)
            elif key == "action":
                # Map back from RoboDM convention to Robomimic 'actions'
                demo_grp.create_dataset("actions", data=value)
            elif key in ["rewards", "dones", "states"]:
                demo_grp.create_dataset(key, data=value)
            elif key.startswith("metadata/"):
                # Metadata already handled above
                continue
            else:
                # Any other unexpected keys
                if key not in ["action", "rewards", "dones", "states"]:
                   demo_grp.create_dataset(key, data=value)

        demo_grp.attrs["num_samples"] = num_samples
        total_samples += num_samples

    # Finalize global metadata
    data_grp.attrs["total"] = total_samples
    f_out.close()
    print(f"Conversion complete. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla_dir", type=str, required=True, help="Directory containing .vla files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save robomimic hdf5 dataset")
    args = parser.parse_args()
    convert_vla_to_hdf5(args.vla_dir, args.output_path)
