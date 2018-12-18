#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(
    description="Create a numeric index of classes for 20BN datasets",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("labels_csv", type=Path, help="Path to labels CSV file, "
                                                  "e.g. something-something-v1-labels.csv")
parser.add_argument("numeric_labels_csv", type=Path, help="Path to output CSV containing mapping between "
                                                          "named categories and their numeric labels")


def read_labels(labels_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, header=None, names=["name"])
    # We sort the labels to be consistent with the original
    # TRN-pytorch implementation that does (so we can compare models)
    labels.sort_values(by='name', inplace=True)
    labels.index.name = "id"
    labels = labels.reset_index()
    return labels


def main(args):
    labels_path = args.labels_csv
    labels = read_labels(labels_path)
    labels.set_index('name', inplace=True)
    labels.to_csv(args.numeric_labels_csv)


if __name__ == '__main__':
    main(parser.parse_args())
