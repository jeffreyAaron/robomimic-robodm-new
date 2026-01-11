import torch
import numpy as np
import wandb
import io
import traceback
from PIL import Image
import matplotlib.pyplot as plt

import robomimic.utils.tensor_utils as TensorUtils

class SaliencyWrapper:
    def __init__(self, algo):
        self.algo = algo
        self.vis_batch = None
        self.last_log_epoch = -1

    def __getattr__(self, name):
        return getattr(self.algo, name)

    def train_on_batch(self, batch, epoch, validate=False):
        # Capture the first training batch for visualization
        if self.vis_batch is None and not validate:
            # Clone the batch to avoid issues with the original batch being modified
            # We use a custom clone to handle None values correctly
            def safe_detach(x):
                return TensorUtils.recursive_dict_list_tuple_apply(x, {
                    torch.Tensor: lambda t: t.detach(),
                    type(None): lambda n: n,
                })
            
            def safe_clone(x):
                return TensorUtils.recursive_dict_list_tuple_apply(x, {
                    torch.Tensor: lambda t: t.clone(),
                    np.ndarray: lambda a: a.copy(),
                    type(None): lambda n: n,
                })

            self.vis_batch = safe_detach(safe_clone(batch))
            print(f"Captured visualization batch with keys: {list(self.vis_batch['obs'].keys())}")
        
        return self.algo.train_on_batch(batch, epoch, validate=validate)

    def on_epoch_end(self, epoch):
        self.algo.on_epoch_end(epoch)
        
        # Log visualizations every 10 epochs for testing, then every 50
        freq = 10 if epoch <= 50 else 50
        if self.vis_batch is not None and epoch % freq == 0 and epoch != self.last_log_epoch:
            self.last_log_epoch = epoch
            print(f"--- Attempting to log visualizations at epoch {epoch} ---")
            self.log_raw_images(epoch)
            self.log_saliency(epoch)
            self.log_fft_spectrum(epoch)

    def log_raw_images(self, epoch):
        if wandb.run is None:
            return

        batch = self.vis_batch
        vis_keys = [k for k in batch["obs"] if "image" in k]
        if not vis_keys:
            return

        log_dict = {}
        for k in vis_keys:
            img_tensor = batch["obs"][k]
            # Get first sample in batch
            img_idx = 0
            
            # If sequence [B, T, C, H, W], get first timestamp (index 0)
            if len(img_tensor.shape) == 5:
                img = img_tensor[img_idx, 0].detach().cpu().numpy()
            else:
                img = img_tensor[img_idx].detach().cpu().numpy()

            # Transpose to [H, W, C]
            img = np.transpose(img, (1, 2, 0))
            
            # Normalize for display
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
            
            log_dict[f"raw_images/{k}"] = wandb.Image(img, caption=f"Epoch {epoch} (Batch 0, T0)")

        if log_dict:
            print(f"Logging {len(log_dict)} raw observation frames to wandb")
            wandb.log(log_dict, step=epoch)

    def log_fft_spectrum(self, epoch):
        if wandb.run is None:
            return

        batch = self.vis_batch
        vis_keys = [k for k in batch["obs"] if "image" in k]
        if not vis_keys:
            return

        log_dict = {}
        for k in vis_keys:
            # Get first image in batch
            img_idx = 0
            img_tensor = batch["obs"][k]
            
            # Handle sequence dimension [B, T, C, H, W] vs [B, C, H, W]
            if len(img_tensor.shape) == 5:
                img = img_tensor[img_idx, -1].detach().cpu().numpy()
            else:
                img = img_tensor[img_idx].detach().cpu().numpy()

            # Average over channels for FFT (Grayscale)
            img_gray = np.mean(img, axis=0)

            # Compute 2D FFT
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            
            # Power Spectrum (log scale for better visualization)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            
            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            im = ax.imshow(magnitude_spectrum, cmap='magma')
            ax.set_title(f"FFT Power Spectrum - {k}")
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            
            # Convert to PIL
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_pil = Image.open(buf)
            plt.close(fig)

            log_dict[f"fft_spectrum/{k}"] = wandb.Image(img_pil, caption=f"Epoch {epoch}")

        if log_dict:
            print(f"Logging {len(log_dict)} FFT spectrum plots to wandb")
            wandb.log(log_dict, step=epoch)

    def log_saliency(self, epoch):
        if wandb.run is None:
            print("Warning: wandb.run is None in log_saliency")
            return

        # Prepare batch
        # Use safe clone to handle None values
        def safe_clone(x):
            return TensorUtils.recursive_dict_list_tuple_apply(x, {
                torch.Tensor: lambda t: t.clone(),
                np.ndarray: lambda a: a.copy(),
                type(None): lambda n: n,
            })
        
        batch = safe_clone(self.vis_batch)
        vis_keys = [k for k in batch["obs"] if "image" in k]
        if not vis_keys:
            print(f"No image keys found in batch obs: {list(batch['obs'].keys())}")
            return

        print(f"Computing saliency for keys: {vis_keys}")
        # Enable grads
        for k in vis_keys:
            batch["obs"][k].requires_grad = True

        # Forward pass
        self.algo.nets.train() # Ensure we're in train mode for grads
        
        # Depending on the algo, we might need different forward calls
        try:
            if hasattr(self.algo, "_forward_training"):
                predictions = self.algo._forward_training(batch)
            else:
                # Fallback for other algos
                predictions = {"actions": self.algo.nets["policy"](batch["obs"])}
            
            actions = predictions.get("actions", None)
            if actions is None:
                print("No actions found in predictions")
                return

            # Backward pass for saliency
            loss = actions.abs().sum()
            loss.backward()

            log_dict = {}
            for k in vis_keys:
                grad = batch["obs"][k].grad
                if grad is None:
                    print(f"Gradient is None for key {k}")
                    continue
                
                # grad shape can be [B, C, H, W] or [B, T, C, H, W]
                grad_abs = grad.abs()
                if len(grad_abs.shape) == 5: # [B, T, C, H, W]
                    grad_abs = grad_abs[:, -1]
                    orig_img = batch["obs"][k][:, -1]
                else:
                    orig_img = batch["obs"][k]

                # Mean over channels
                saliency = grad_abs.mean(dim=1).detach().cpu().numpy()
                
                # Take the first image in the batch for visualization
                img_idx = 0
                s_map = saliency[img_idx]
                if s_map.max() > s_map.min():
                    s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min())
                
                img = orig_img[img_idx].detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0)) # [H, W, C]
                if img.max() > img.min():
                    img = (img - img.min()) / (img.max() - img.min())

                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(img)
                ax.imshow(s_map, cmap='jet', alpha=0.5)
                ax.axis('off')
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                img_pil = Image.open(buf)
                plt.close(fig)

                log_dict[f"saliency/{k}"] = wandb.Image(img_pil, caption=f"Epoch {epoch}")

            if log_dict:
                print(f"Logging {len(log_dict)} saliency maps to wandb (run id: {wandb.run.id})")
                wandb.log(log_dict, step=epoch)
            else:
                print("log_dict is empty")

        except Exception as e:
            print(f"Failed to compute saliency map: {e}")
            traceback.print_exc()
