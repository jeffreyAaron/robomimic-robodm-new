import os
import subprocess
import wandb

class DatasetManager:
    def __init__(self, task, crf):
        self.task = task
        self.crf = crf
        self.base_dataset = f"datasets/{task}/ph/image_v15.hdf5"
        self.vla_dir = f"datasets/{task}/ph/image_v15_vla_crf{crf}"
        self.reconstructed_hdf5 = f"datasets/{task}/ph/image_v15_reconstructed_crf{crf}.hdf5"

    def prepare_dataset(self):
        """
        Check if reconstructed dataset exists, if not, run compression and reconstruction.
        """
        # Set environment for subprocess to include robodm
        env = os.environ.copy()
        robodm_path = os.path.abspath("robodm")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{robodm_path}"
        else:
            env["PYTHONPATH"] = robodm_path

        if not os.path.exists(self.reconstructed_hdf5):
            print(f"Reconstructed dataset not found at {self.reconstructed_hdf5}. Generating for CRF {self.crf}...")
            
            # Robomimic to VLA (compression)
            cmd_to_vla = [
                "python", "robomimic_to_vla_compressed.py",
                "--dataset", self.base_dataset,
                "--output_dir", self.vla_dir,
                "--crf", str(self.crf)
            ]
            print(f"Running: {' '.join(cmd_to_vla)}")
            subprocess.run(cmd_to_vla, check=True, env=env)
            
            # VLA to Robomimic (reconstruction)
            cmd_from_vla = [
                "python", "vla_to_robomimic.py",
                "--vla_dir", self.vla_dir,
                "--output_path", self.reconstructed_hdf5
            ]
            print(f"Running: {' '.join(cmd_from_vla)}")
            subprocess.run(cmd_from_vla, check=True, env=env)
            
            # Calculate and print directory size to verify compression
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.vla_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            vla_size_mb = total_size / (1024 * 1024)
            print(f"Total VLA directory size (CRF {self.crf}): {vla_size_mb:.2f} MB")
            
            # Log VLA size to WandB
            if wandb.run is not None:
                wandb.log({"vla_dataset_size_mb": vla_size_mb})
                wandb.run.summary["vla_dataset_size_mb"] = vla_size_mb
            
            print(f"Intermediate VLA files kept at: {self.vla_dir}")
        else:
            print(f"Using existing reconstructed dataset at {self.reconstructed_hdf5}")

        return self.reconstructed_hdf5
