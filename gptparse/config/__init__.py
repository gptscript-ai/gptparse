import json
import os

CONFIG_FILE = os.path.expanduser("~/.gptparse_config.json")


def get_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def set_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def print_config():
    config = get_config()
    if not config:
        print("No configuration set.")
    else:
        print("Current configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")


# Include the setup_logging function here as well
from .logging_config import setup_logging
