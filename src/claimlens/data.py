import json
from pathlib import Path

import pandas as pd

#This should point at your root on your own computer (dont need to change)
ROOT = Path(__file__).resolve().parents[2]                  # data.py -> claimlens -> src -> ClaimLens (repo folder)
DEFAULT_PATH = ROOT / "raw_data" / "abcd_full.json"         # the "/" sticks folder names together


def load_data(path=DEFAULT_PATH) -> pd.DataFrame:
    """Load all ABCD conversations into one DataFrame, with a 'split' column."""
    # open the json and turn it into a python dict: {"train": [...], "dev": [...], "test": [...]}
    with open(path) as f:
        data = json.load(f)

    # make one df per part (train/dev/test) and stack them into one big df
    df = pd.concat(
        [pd.DataFrame(rows).assign(split=name)    # one table per part + a "split" column
         for name, rows in data.items()],         # loop over ("train", [...]), ("dev", [...]), ...
        ignore_index=True,                        # renumber rows 0, 1, 2... across all parts
    )
    return df
