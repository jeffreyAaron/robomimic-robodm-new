import os
import wandb
import torch
import numpy as np
from datetime import datetime

import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train
from robomimic.scripts.run_trained_agent import run_trained_agent

from custom.wrappers.saliency_wrapper import SaliencyWrapper
from custom.utils.eval_utils import EvalArgs
from custom.utils.dataset_utils import DatasetManager

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
    seed = getattr(config, "seed", 42)
    
    print(f"Starting run for task: {task}, CRF: {crf}, Seed: {seed}")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 2. Dataset Preparation
    dataset_manager = DatasetManager(task, crf)
    reconstructed_hdf5 = dataset_manager.prepare_dataset()

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
    
    # Configure Image-based training
    # 1. Enable RGB modalities
    robomimic_config.observation.modalities.obs.rgb = ["agentview_image", "robot0_eye_in_hand_image"]
    
    # 2. Configure Low-dim modalities (removing 'object' as requested)
    robomimic_config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos", 
        "robot0_eef_quat", 
        "robot0_gripper_qpos"
    ]
    
    # 3. Increase data workers for image loading
    robomimic_config.train.num_data_workers = 2
    
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
    
    # Monkey-patch algo_factory to wrap the model with SaliencyWrapper
    import robomimic.algo as Algo
    import robomimic.scripts.train as TrainScript
    original_algo_factory = Algo.algo_factory

    def patched_algo_factory(*args, **kwargs):
        algo = original_algo_factory(*args, **kwargs)
        print(f"Creating SaliencyWrapper for algo {type(algo)}")
        return SaliencyWrapper(algo)

    Algo.algo_factory = patched_algo_factory
    TrainScript.algo_factory = patched_algo_factory

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
