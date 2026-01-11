class EvalArgs:
    """
    Helper class to mimic argparse results for run_trained_agent.
    """
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
