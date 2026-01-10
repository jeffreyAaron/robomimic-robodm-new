import os
import argparse
import wandb
import torch
import numpy as np
import subprocess
import json
from datetime import datetime

import robomimic
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train
from robomimic.scripts.run_trained_agent import run_trained_agent

def main():
    # 1. Setup WandB Sweep logic
    # Ensure model files are not uploaded to WandB
    os.environ["WANDB_IGNORE_GLOBS"] = "*.pth"
    
    # When running as part of a sweep, wandb.init() will pick up the sweep config
    run = wandb.init()
    
    # HACK: Robomimic's internal logger requires WANDB_ENTITY to be set in its macros.
    # We set it here programmatically from the active wandb run.
    import robomimic.macros as Macros
    Macros.WANDB_ENTITY = run.entity
    
    # Configuration
    # If running in a sweep, wandb.config will be populated with sweep params
    config = wandb.config
    
    # Task and parameters - defaults if not in sweep
    task = getattr(config, "task", "can")
    crf = getattr(config, "crf", 23)
    num_epochs = getattr(config, "num_epochs", 500)
    batch_size = getattr(config, "batch_size", 256)
    
    print(f"Starting run for task: {task}, CRF: {crf}")
    
    # 2. Dataset Preparation
    # We assume the base dataset exists at datasets/{task}/ph/image_v15.hdf5
    base_dataset = f"datasets/{task}/ph/image_v15.hdf5"
    vla_dir = f"datasets/{task}/ph/image_v15_vla_crf{crf}"
    reconstructed_hdf5 = f"datasets/{task}/ph/image_v15_reconstructed_crf{crf}.hdf5"
    
    # Set environment for subprocess to include robodm
    env = os.environ.copy()
    robodm_path = os.path.abspath("robodm")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{robodm_path}"
    else:
        env["PYTHONPATH"] = robodm_path

    if not os.path.exists(reconstructed_hdf5):
        print(f"Reconstructed dataset not found at {reconstructed_hdf5}. Generating for CRF {crf}...")
        
        # Robomimic to VLA (compression)
        cmd_to_vla = [
            "python", "robomimic_to_vla_compressed.py",
            "--dataset", base_dataset,
            "--output_dir", vla_dir,
            "--crf", str(crf)
        ]
        print(f"Running: {' '.join(cmd_to_vla)}")
        subprocess.run(cmd_to_vla, check=True, env=env)
        
        # VLA to Robomimic (reconstruction)
        cmd_from_vla = [
            "python", "vla_to_robomimic.py",
            "--vla_dir", vla_dir,
            "--output_path", reconstructed_hdf5
        ]
        print(f"Running: {' '.join(cmd_from_vla)}")
        subprocess.run(cmd_from_vla, check=True, env=env)
        
        # Calculate and print directory size to verify compression
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(vla_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        vla_size_mb = total_size / (1024 * 1024)
        print(f"Total VLA directory size (CRF {crf}): {vla_size_mb:.2f} MB")
        
        # Log VLA size to WandB
        run.log({"vla_dataset_size_mb": vla_size_mb})
        run.summary["vla_dataset_size_mb"] = vla_size_mb
        
        print(f"Intermediate VLA files kept at: {vla_dir}")
        # shutil.rmtree(vla_dir)
    else:
        print(f"Using existing reconstructed dataset at {reconstructed_hdf5}")

    # 3. Training Setup
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Logical naming for the experiment
    run_name = f"{task}_crf{crf}_{timestamp}"
    # Set output_dir to 'test_output' so robomimic creates 'test_output/{run_name}/{timestamp}'
    output_parent_dir = os.path.abspath("test_output")
    
    # Load default BC config
    robomimic_config = config_factory(algo_name="bc")
    
    # Update config with parameters
    robomimic_config.experiment.name = run_name
    robomimic_config.train.output_dir = output_parent_dir
    robomimic_config.train.data = [{"path": reconstructed_hdf5}]
    robomimic_config.train.batch_size = batch_size
    robomimic_config.train.num_epochs = num_epochs
    
    # Enable WandB in Robomimic internal logger
    robomimic_config.experiment.logging.log_wandb = True
    robomimic_config.experiment.logging.wandb_proj_name = run.project
    
    # Enable rollouts during training
    robomimic_config.experiment.rollout.enabled = True
    robomimic_config.experiment.rollout.rate = 100 
    
    # Algo settings (matching simple_test.py)
    robomimic_config.algo.gmm.enabled = False
    
    # Get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # 4. Launch Training
    print(f"Starting training. Experiment: {run_name}")
    
    # HACK: Robomimic's training script calls wandb.finish() at the end, 
    # which would close the run before we can log evaluation results.
    # We temporarily monkeypatch wandb.finish to be a no-op.
    original_finish = wandb.finish
    wandb.finish = lambda *args, **kwargs: None
    
    try:
        train(robomimic_config, device=device)
    finally:
        # Restore original finish
        wandb.finish = original_finish
    
    # 5. Evaluation
    print("Training finished. Starting evaluation...")
    
    # Robomimic creates a directory structure: {output_parent_dir}/{run_name}/{timestamp}/models
    # We use glob to find the checkpoint since we don't know the exact internal timestamp
    import glob
    ckpt_pattern = os.path.join(output_parent_dir, run_name, "*", "models", f"model_epoch_{num_epochs}.pth")
    matching_ckpts = glob.glob(ckpt_pattern)
    
    if not matching_ckpts:
        print(f"Warning: Could not find exact checkpoint {ckpt_pattern}")
        # Try to find the latest model in any subdirectory
        ckpt_pattern_any = os.path.join(output_parent_dir, run_name, "*", "models", "*.pth")
        matching_ckpts = sorted(glob.glob(ckpt_pattern_any))
        if matching_ckpts:
            checkpoint_path = matching_ckpts[-1]
            print(f"Using latest found checkpoint: {checkpoint_path}")
        else:
            print(f"Error: No checkpoints found in {os.path.join(output_parent_dir, run_name)}")
            run.finish()
            return
    else:
        # Use the first match (there should usually be only one timestamp folder unless training was resumed)
        checkpoint_path = matching_ckpts[-1]

    print(f"Evaluating checkpoint: {checkpoint_path}")

    # Helper class to mimic argparse results for run_trained_agent
    class EvalArgs:
        def __init__(self, agent, n_rollouts=50, horizon=None, env=None, render=False, 
                     video_path=None, video_skip=5, camera_names=None, dataset_path=None, 
                     dataset_obs=False, seed=0):
            self.agent = agent
            self.n_rollouts = n_rollouts
            self.horizon = horizon
            self.env = env
            self.render = render
            self.video_path = video_path
            self.video_skip = video_skip
            self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
            self.dataset_path = dataset_path
            self.dataset_obs = dataset_obs
            self.seed = seed

    # Standard evaluation: 50 rollouts
    eval_args = EvalArgs(agent=checkpoint_path, n_rollouts=50)
    
    # Run evaluation (using the modified run_trained_agent that returns stats)
    eval_stats = run_trained_agent(eval_args)
    
    # Log evaluation results to WandB
    if eval_stats:
        print("Logging evaluation stats to WandB...")
        formatted_stats = {f"eval/{k}": v for k, v in eval_stats.items()}
        wandb.log(formatted_stats)
        
        # Update summary for easy comparison in the WandB table
        for k, v in formatted_stats.items():
            run.summary[k] = v
            
    print(f"Run {run_name} completed successfully.")
    run.finish()

if __name__ == "__main__":
    main()
