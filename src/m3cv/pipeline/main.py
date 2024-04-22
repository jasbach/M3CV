from argparse import ArgumentParser

from m3cv.ConfigHandler import Config
from m3cv.pipeline.io import DataLoader

"""Script that houses primary entry point for model training runs.
"""

def run(args):
    #parser = ArgumentParser()
    #parser.add_argument('configpath')
    args = parser.parse_args(args)
    with open(args.configpath, 'r') as f:
        config = Config(f)

    loader = DataLoader(config)


if __name__ == '__main__':
    run([r'F:\repos\M3CV\src\m3cv\templates\config_outcome_prediction.yaml'])