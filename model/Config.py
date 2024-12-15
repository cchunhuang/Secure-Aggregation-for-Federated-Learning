import json

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
            
def loadConfig(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Config(config_dict)