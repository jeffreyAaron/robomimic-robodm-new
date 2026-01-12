from robomimic.config import config_factory

class BaseAlgoConfig:
    """Base class for all algorithm configurations used in sweeps."""
    
    def __init__(self, algo_name):
        self.config = config_factory(algo_name=algo_name)

    def configure(self, run_name, output_dir, dataset_path, batch_size=256, num_epochs=500, wandb_proj_name="debug"):
        # Unlock config to allow modifications
        self.config.unlock()

        # Experiment settings
        self.config.experiment.name = run_name
        self.config.train.output_dir = output_dir
        self.config.train.data = [{"path": dataset_path}]
        
        # Common parameters from Drolet et al. / Robomimic standards
        self.config.train.batch_size = batch_size # Standard robust batch size (Drolet suggests 1024 for MLP, but 256 is safer for RNN/Trans).
        self.config.train.num_epochs = num_epochs # Ensure convergence on harder tasks.
        self.config.experiment.save.enabled = True 
        
        # Observation modalities
        self.config.observation.modalities.obs.rgb = ["agentview_image", "robot0_eye_in_hand_image"]
        self.config.observation.modalities.obs.low_dim = [
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos"
        ]
        
        # Performance
        self.config.train.num_data_workers = 2
        
        # Logging
        self.config.experiment.logging.log_wandb = True
        self.config.experiment.logging.wandb_proj_name = wandb_proj_name
        
        # Rollouts
        self.config.experiment.rollout.enabled = True
        self.config.experiment.rollout.rate = 100
        
        # Call algo-specific configuration
        self.set_algo_parameters()
        
        # Lock config again after modifications
        self.config.lock()
        
        return self.config

    def set_algo_parameters(self):
        """Override this in subclasses to set specific algorithm parameters."""
        pass


class BCConfig(BaseAlgoConfig):
    """Standard BC (MLP) configuration."""
    def __init__(self):
        super().__init__(algo_name="bc")

    def set_algo_parameters(self):
        # 0. BC (MLP)
        self.config.algo.rnn.enabled = False
        self.config.algo.gmm.enabled = False


class BCRNNConfig(BaseAlgoConfig):
    """BC-RNN configuration based on Drolet et al. suggestions."""
    def __init__(self):
        super().__init__(algo_name="bc")

    def set_algo_parameters(self):
        # 1. BC-RNN
        self.config.algo.rnn.enabled = True
        self.config.algo.rnn.horizon = 20          # Drolet et al. suggest 10 is too short for precision; 20 handles occlusion better.
        self.config.algo.rnn.hidden_dim = 400      # Robomimic standard.
        self.config.algo.rnn.num_layers = 2        # Number of RNN layers that are stacked.
        
        self.config.algo.gmm.enabled = True        # Crucial: Gaussian Mixture Models (GMM) handle "jitter" much better than MSE.
        self.config.algo.gmm.num_modes = 5         # Drolet Optimal: Enough modes for complex motion, not enough to overfit.


class BCTransformerConfig(BaseAlgoConfig):
    """BC-Transformer configuration."""
    def __init__(self):
        super().__init__(algo_name="bc")

    def set_algo_parameters(self):
        # 2. BC-Transformer
        self.config.algo.transformer.enabled = True
        self.config.algo.transformer.context_length = 10 # Hypothesis: You will likely need to increase this to 30 for compressed data.
        self.config.algo.transformer.embed_dim = 256     # Drolet setting (Robomimic default is 512, but 256 prevents overfitting on smaller datasets).
        self.config.algo.transformer.num_layers = 4
        self.config.algo.transformer.num_heads = 4


class DiffusionConfig(BaseAlgoConfig):
    """Diffusion Policy configuration based on Chi et al. (2023)."""
    def __init__(self):
        super().__init__(algo_name="diffusion_policy")

    def set_algo_parameters(self):
        # 3. Diffusion Policy
        self.config.algo.ddpm.num_inference_timesteps = 100 # Chi et al. (2023) Standard. High steps = high precision.
        self.config.algo.horizon.prediction_horizon = 16   # Lookahead window.
        self.config.algo.horizon.observation_horizon = 2    # Uses 2 frames of history.
        self.config.algo.horizon.action_horizon = 8         # Executes 8 steps before re-planning (Receding Horizon Control).


class CQLConfig(BaseAlgoConfig):
    """CQL (Conservative Q-Learning) configuration."""
    def __init__(self):
        super().__init__(algo_name="cql")

    def set_algo_parameters(self):
        # 4. CQL
        self.config.algo.critic.cql_weight = 1.0     # The "Conservatism" weight. If this is too high on compressed data, the robot will freeze.
        self.config.algo.critic.target_q_gap = 5.0  # Alternative tuning knob for conservatism.
        self.config.algo.optim_params.actor.learning_rate.initial = 1e-4         # Slower actor learning stabilizes Offline RL.
        self.config.algo.optim_params.critic.learning_rate.initial = 3e-4        # Critic learns faster to guide the actor.


class IQLConfig(BaseAlgoConfig):
    """IQL (Implicit Q-Learning) configuration."""
    def __init__(self):
        super().__init__(algo_name="iql")

    def set_algo_parameters(self):
        # 5. IQL
        self.config.algo.vf_quantile = 0.7     # Standard for "Proficient Human" data.
        self.config.algo.adv.beta = 3.0   # Controls how much advantage weights the policy update.


# Registry to easily fetch config classes by name
ALGO_CONFIG_REGISTRY = {
    "bc": BCConfig,
    "bc_rnn": BCRNNConfig,
    "bc_transformer": BCTransformerConfig,
    "diffusion_policy": DiffusionConfig,
    "cql": CQLConfig,
    "iql": IQLConfig,
}

def get_config_by_name(name):
    if name not in ALGO_CONFIG_REGISTRY:
        raise ValueError(f"Algorithm config '{name}' not found in ALGO_CONFIG_REGISTRY. "
                         f"Available: {list(ALGO_CONFIG_REGISTRY.keys())}")
    return ALGO_CONFIG_REGISTRY[name]()
