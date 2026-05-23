"""Hydra-friendly utility helpers for the pixel-supervision ablation.

This is the standard lightning-hydra-template helper layer, kept
verbatim so existing experiments continue to work. Highlights:

* :func:`instantiate_callbacks`, :func:`instantiate_loggers` — turn a
  Hydra ``DictConfig`` of ``_target_`` entries into Lightning objects.
* :func:`task_wrapper` — wraps the main task in try/finally with
  metric extraction + exception logging.
* :class:`RankedLogger` — rank-zero-aware logger.
* :func:`print_config_tree`, :func:`enforce_tags` — rich-formatted
  config display + tag-prompt for unnamed runs.
"""

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.utils import extras, get_metric_value, task_wrapper

__all__ = [
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "RankedLogger",
    "enforce_tags",
    "print_config_tree",
    "extras",
    "get_metric_value",
    "task_wrapper",
]
