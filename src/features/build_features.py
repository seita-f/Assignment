import yaml
import pandas as pd
from pathlib import Path


def main():
    # load config
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
   
if __name__ == "__main__":
    main()