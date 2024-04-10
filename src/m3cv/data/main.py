from argparse import ArgumentParser

from m3cv.ConfigHandler import Config

"""Script that houses the primary entrypoint for performing data packing.
"""

def run():
    parser = ArgumentParser()
    parser.add_argument('configpath')
    args = parser.parse_args()
    with open(args.configpath, 'r') as f:
        config = Config(f)

    