from .eval import eval
from .train import train
from .learncurve import learning_curve
from .predict import predict
from .prep import prep
from .train_checkpoint import train_checkpoint

COMMAND_FUNCTION_MAP = {
    "prep": prep,
    "train": train,
    "eval": eval,
    "predict": predict,
    "learncurve": learning_curve,
    "train_checkpoint": train_checkpoint,
}

CLI_COMMANDS = tuple(COMMAND_FUNCTION_MAP.keys())


def cli(command, config_file):
    """command-line interface

    Parameters
    ----------
    command : string
        One of {'prep', 'train', 'eval', 'predict', 'learncurve', 'train_checkpoint'}
    config_file : str, Path
        path to a config.toml file
    """
    if command in COMMAND_FUNCTION_MAP:
        COMMAND_FUNCTION_MAP[command](toml_path=config_file)
    else:
        raise ValueError(f"command not recognized: {command}")
