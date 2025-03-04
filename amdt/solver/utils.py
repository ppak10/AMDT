import configparser
import os

from amdt import data
from importlib.resources import files


class SolverUtils:
    """
    Class for handling solver utility functions
    """

    @staticmethod
    def load_config_file(config_dir, config_file, config_override):
        """
        Loads configs from prescribed file and also applies given overrides.
        """

        config = configparser.ConfigParser()
        config_file_path = os.path.join(config_dir, config_file)
        config_resource = files(data).joinpath(config_file_path)
        config.read(config_resource)
        output = {}

        for section in config.sections():
            for key, value in config[section].items():
                if section == "float":
                    output[key] = float(value)
                else:
                    # Defaults to string
                    output[key] = value

        for key, value in config_override.items():
            output[key] = value

        return output
