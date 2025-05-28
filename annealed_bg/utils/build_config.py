import omegaconf
from omegaconf import DictConfig, OmegaConf

from annealed_bg.config.main import Config


def build_config(cfg_dict: DictConfig) -> tuple[Config, dict]:
    """Build the config object by parsing the config dict.

    Args:
        cfg_dict: The config dict.

    Returns:
        The config object and the config dict.
    """

    OmegaConf.register_new_resolver("eval", eval)

    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg_dict, resolve=True, throw_on_missing=True
    )

    cfg: Config = Config(**cfg_dict)

    return cfg, cfg_dict
