import importlib
import yaml
import json
from os.path import join


class Config(dict):
    """Config class that allows for dot notation."""

    def __init__(self, dictionary={}):
        super().__init__(dictionary)
        for key, value in self.items():
            self.__setitem__(key, value)
            self.__setattr__(key, value)

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = Config(value)
        elif isinstance(value, str):
            value = self.str_to_num(value)
        super().__setattr__(key, value)
        super().__setitem__(key, value)

    def str_to_num(self, s):
        """Converts a string to a float or int if possible."""
        try:
            return float(s)
        except ValueError:
            return s

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Config(value)
        if isinstance(value, str):
            value = self.str_to_num(value)
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, name):
        if name in self:
            dict.__delitem__(
                self, name
            )  # Use the parent class's method to avoid recursion
        if hasattr(self, name):
            super().__delattr__(name)

    def __delitem__(self, name):
        if name in self:
            dict.__delitem__(
                self, name
            )  # Use the parent class's method to avoid recursion
        if hasattr(self, name):
            super().__delattr__(name)

    def yaml_repr(self, dumper):
        return dumper.represent_dict(self.to_dict())

    def to_dict(self):
        """Converts the object to a dictionary, including any attributes."""
        result = {}
        for key, value in self.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def save_to_yaml(config, file_name):
        with open(file_name, "w") as file:
            yaml.dump(config.to_dict(), file)

    def save_pretrained(self, folder: str):
        """
        Saves the config to a json file.
        For compatibility with trainer.
        """
        file_name = join(folder, "model_config.json")
        with open(file_name, "w") as file:
            json.dump(self.to_dict(), file)

    def update(self, config: "Config"):
        """Updates the config with a different config. Update only if key is not present in self."""
        for key, value in config.items():
            if isinstance(value, dict):
                value = Config(value)
            if key not in self:
                setattr(self, key, value)

def instantiate_class(instantiate_config, **extra_kwargs):
    """Instantiates a class from a config object."""
    module_path, class_name = instantiate_config._target_.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    kwargs = {k: v for k, v in instantiate_config.items() if k != "_target_"}
    # Merge config kwargs with extra kwargs
    kwargs.update(extra_kwargs)
    instance = class_(**kwargs)
    return instance


def instantiate_function(func_path: str):
    """Initializes a function or a class static method from a path string."""
    parts = func_path.rsplit(
        ".", 2
    )  # Split into module, class (optional), and function/method

    if len(parts) == 3:
        # If there are three parts, it includes a class
        module_path, class_name, func_name = parts
        module = importlib.import_module(module_path)
        klass = getattr(module, class_name)
        # Check if klass is a class and func_name is a static method
        if isinstance(klass, type) and hasattr(klass, func_name):
            method = getattr(klass, func_name)
            if callable(method):
                return method
        else:
            raise ValueError(f"{func_name} is not a static method of {class_name}")
    elif len(parts) == 2:
        # If there are two parts, it's a function in a module
        module_path, func_name = parts
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    else:
        raise ValueError(
            "Function path must be in the format 'module.Class.method' or 'module.function'")

def load_config(config_file):
    """Loads a yaml config file."""
    with open(config_file, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    cfg = Config(cfg)
    return cfg
