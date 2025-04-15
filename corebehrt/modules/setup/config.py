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


def instantiate_function(instantiate_config, **extra_kwargs):
    """
    Initializes a function or a class static method from a config object.
    The config must include a '_target_' key specifying the full path to the function.
    """
    if "_target_" not in instantiate_config:
        raise ValueError("The configuration must include a '_target_' key.")

    func_path = instantiate_config["_target_"]
    parts = func_path.rsplit(
        ".", 2
    )  # Split into module, submodule (optional), and function/method

    if len(parts) == 3:
        # If there are three parts, it includes a submodule or class
        module_path, submodule, func_name = parts
        module = importlib.import_module(module_path)
        submodule_or_class = getattr(module, submodule, None)

        if submodule_or_class is None:
            raise ValueError(f"{submodule} is not found in module {module_path}")

        target = submodule_or_class

    elif len(parts) == 2:
        # If there are two parts, it's a function in a module
        module_path, func_name = parts
        module = importlib.import_module(module_path)
        target = module
    else:
        raise ValueError(
            "Function path must be in the format 'module.submodule.function', 'module.Class.method', or 'module.function'"
        )

    func = getattr(target, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"{func_name} is not a callable function in {target}")

    # Merge config kwargs with extra kwargs
    kwargs = {k: v for k, v in instantiate_config.items() if k != "_target_"}
    kwargs.update(extra_kwargs)

    # If the function accepts arguments, pass them
    return lambda *args, **kwargs_: func(*args, **{**kwargs, **kwargs_})


def load_config(config_file):
    """Loads a yaml config file."""
    with open(config_file, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    cfg = Config(cfg)
    return cfg
