import os
os.environ["MUJOCO_GL"] = "osmesa"
import shutil
import robomimic
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

# make default BC config
config = config_factory(algo_name="bc")

TASK="can"

# set config attributes here that you would like to update
config.experiment.name = f"test_{TASK}_simple"
output_dir = os.path.abspath("test_output")
# if os.path.exists(output_dir):
#     shutil.rmtree(output_dir)
config.train.output_dir = output_dir
config.train.data = [{"path": f"datasets/{TASK}/ph/image_v15_reconstructed.hdf5"}]
config.train.batch_size = 256
config.train.num_epochs = 500
config.experiment.rollout.enabled = True # we want to test rendering
config.experiment.rollout.rate = 100 # run rollouts every epoch
config.algo.gmm.enabled = False

# get torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# launch training run
train(config, device=device)


