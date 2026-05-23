"""Miscellaneous Hydra-task helpers: ``extras``, ``task_wrapper``, ``get_metric_value``."""
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value



def extract_cuboids(data, t, h, w, overlap_t=0.3, overlap_h=0.5, overlap_w=0.5):
    """
    Extracts overlapping cuboids from a 3D numpy array."""
    T, H, W = data.shape

    # Compute stride for each dimension based on overlap
    stride_t = int(t * (1 - overlap_t))
    stride_h = int(h * (1 - overlap_h))
    stride_w = int(w * (1 - overlap_w))

    cuboids = []
    hw_origins = []
    for t_start in range(0, T - t + 1, stride_t):
        for h_start in range(0, H - h + 1, stride_h):
            for w_start in range(0, W - w + 1, stride_w):
                cuboid = data[t_start:t_start + t, h_start:h_start + h, w_start:w_start + w]
                cuboids.append(cuboid)
                hw_origins.append((h_start, w_start))

    return np.array(cuboids), np.array(hw_origins)



def load_data_from_h5py(file_path):
        """
        Load data from HDF5 file in a memory-efficient way.
        Labels are NOT duplicated across augmentations.
        
        Returns:
            latents_by_traj: List of arrays, each (num_aug, T, D)
            pressures_by_traj: List of arrays, each (T, 1)
            winds_by_traj: List of arrays, each (T, 1)
            traj_ids: List of trajectory IDs
        """
        import h5py
        
        latents_by_traj = []
        pressures_by_traj = []
        winds_by_traj = []
        traj_ids = []
        
        with h5py.File(file_path, 'r') as f:
            num_trajectories = f.attrs['num_trajectories']
            feature_dim = f.attrs['feature_dim']
            
            print(f"Loading {num_trajectories} trajectories from {file_path}")
            print(f"Feature dimension: {feature_dim}")
            
            total_frames = 0
            for traj_idx in range(num_trajectories):
                grp = f[f'trajectory_{traj_idx}']
                
                # Load data - labels stored ONCE per trajectory
                latents = grp['latents'][:]  # (T, D)
                pressure = grp['pressure'][:]  # (T, 1)
                wind = grp['wind'][:]  # (T, 1)
                traj_id = grp.attrs['id'] # these are stored as strings
                
                T, D = latents.shape
                total_frames +=  T
                
                latents_by_traj.append(latents)
                pressures_by_traj.append(pressure)
                winds_by_traj.append(wind)
                traj_ids.append(int(traj_id))
        
        print(f"\nLoaded data (memory-efficient):")
        print(f"  Total trajectories: {len(latents_by_traj):,}")
        print(f"  Total frames (all augmentations): {total_frames:,}")
        print(f"  Avg frames per trajectory: {total_frames / num_trajectories:.1f}")
        return latents_by_traj, pressures_by_traj, winds_by_traj, traj_ids
    
    