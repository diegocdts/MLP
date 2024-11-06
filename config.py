import json

def load(file: str, name: str):
    with open(file, 'r') as file:
        config = json.load(file)

    if name in config:
        parameters = config[name]
        return parameters
    else:
        raise ValueError(f'{name} not found in {file} file')