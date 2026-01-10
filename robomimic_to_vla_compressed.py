import os
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm
import robodm
from robomimic.utils.file_utils import get_env_metadata_from_dataset

def convert_dataset(dataset_path, output_dir, num_demos=None, crf=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get environment metadata
    env_meta = get_env_metadata_from_dataset(dataset_path)
    env_meta_json = json.dumps(env_meta)

    f = h5py.File(dataset_path, "r")
    demo_keys = sorted(list(f["data"].keys()))

    if num_demos is not None:
        demo_keys = demo_keys[:num_demos]

    print(f"Converting {len(demo_keys)} trajectories from {dataset_path} to {output_dir}...")
    
    # Prepare codec options
    codec_options = {}
    if crf is not None:
        codec_options["crf"] = str(crf)
        print(f"Using CRF: {crf}")
    else:
        print("Using default CRF (23)")

    for demo_key in tqdm(demo_keys):
        demo_grp = f[f"data/{demo_key}"]
        
        # Prepare data dictionary for Trajectory
        data = {}
        
        # Add observations
        for obs_key in demo_grp["obs"]:
            data[f"observation/{obs_key}"] = demo_grp["obs"][obs_key][()]
        
        # Add other standard keys
        for key in ["actions", "rewards", "dones"]:
            if key in demo_grp:
                # Map names to RoboDM conventions: actions -> action
                target_key = "action" if key == "actions" else key
                data[target_key] = demo_grp[key][()]
        
        # Add next_obs if present
        if "next_obs" in demo_grp:
            for obs_key in demo_grp["next_obs"]:
                data[f"next_observation/{obs_key}"] = demo_grp["next_obs"][obs_key][()]

        # Add states if present
        if "states" in demo_grp:
            data["states"] = demo_grp["states"][()]

        # Determine number of samples
        if "action" in data:
            num_samples = len(data["action"])
        elif "rewards" in data:
            num_samples = len(data["rewards"])
        else:
            # Fallback to first observation
            first_obs = list(demo_grp["obs"].keys())[0]
            num_samples = len(demo_grp["obs"][first_obs])

        # Add metadata as repeated features (to maintain same length for all features)
        # This ensures they are preserved in the VLA format
        data["metadata/env_meta"] = [env_meta_json] * num_samples
        if "model_file" in demo_grp.attrs:
            data["metadata/model_file"] = [demo_grp.attrs["model_file"]] * num_samples
        
        # Create Trajectory
        output_path = os.path.join(output_dir, f"{demo_key}.vla")
        # We use from_dict_of_lists which handles the list-of-arrays format
        robodm.Trajectory.from_dict_of_lists(
            data=data,
            path=output_path,
            video_codec="libx264",
            codec_options=codec_options,
            fps=20 # Default FPS for video encoding
        )

    f.close()
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to robomimic hdf5 dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .vla files")
    parser.add_argument("--num_demos", type=int, default=None, help="Number of trajectories to convert")
    parser.add_argument("--crf", type=int, default=None, help="CRF value for video compression")
    args = parser.parse_args()
    convert_dataset(args.dataset, args.output_dir, args.num_demos, args.crf)
