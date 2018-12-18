#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

parser = argparse.ArgumentParser(
    description="Create file lists for 20BN datasets",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("numeric_labels_csv", type=Path, help="Path to numeric labels CSV file produced by "
                                                          "gen_numeric_labels.py")
parser.add_argument("annotations_csv", type=Path, help="Path to input annotation CSV, "
                                                 "e.g. something-something-v1-train.csv")
parser.add_argument("dataset_root", type=Path, help="Path to dataset root (used for counting frames)")
parser.add_argument("filelist", type=Path, help="Path to output filelist created from input annotation CSV")


def read_annotations(annotation_path: Path) -> pd.DataFrame:
    return pd.read_csv(annotation_path, sep=';', header=None, names=['video_id', 'label'])


def count_frames(annotations: pd.DataFrame) -> None:
    if 'path' not in annotations.columns:
        raise ValueError("Expected annotations to contain 'path' column")

    def _is_img(s: str):
        for ext in ('jpg', 'jpeg', 'png'):
            if s.endswith(ext):
                return True
        return False

    def _count_frames(path: Path):
        return len([c for c in path.iterdir() if _is_img(c.name.lower())])

    annotations['frame_count'] = annotations['path'].progress_apply(_count_frames)


def add_path(annotations: pd.DataFrame, dataset_root: Path) -> None:
    annotations['path'] = annotations['video_id'].apply(lambda id: dataset_root / str(id))


def write_filelist(annotations: pd.DataFrame, file_path: Path):
    with file_path.open('w', encoding='utf8') as f:
        for row in annotations.itertuples():
            line = "{path} {frame_count} {label_id}\n".format(
                    path=row.path,
                    frame_count=row.frame_count,
                    label_id=row.label_id
            )
            f.write(line)


def main(args):
    labels_path = args.numeric_labels_csv
    annotations_path = args.annotations_csv

    # We expect labels to have the format:
    # name,id
    # <label>,<label_id>
    # ...
    labels = pd.read_csv(labels_path, index_col='name')['id']
    # Annotations are of the standard 20BN format:
    # <video_id>,<label>
    annotations = read_annotations(annotations_path)
    annotations['label_id'] = annotations['label'].apply(lambda name: labels.loc[name])
    add_path(annotations, args.dataset_root)
    print("Counting frames")
    count_frames(annotations)
    print("Writing filelist to {}".format(args.filelist))
    write_filelist(annotations, args.filelist)


if __name__ == '__main__':
    main(parser.parse_args())
