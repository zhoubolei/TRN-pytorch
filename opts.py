import logging
import os
from pathlib import Path

import configargparse
from multiprocessing import cpu_count

parser = configargparse.ArgumentParser(
    description="PyTorch implementation of Temporal Segment Networks",
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    auto_env_var_prefix="",  # needed for env var config options to be supported.
)
parser.add_argument("-c", "--config-file", is_config_file=True, help="Config file path")

parser.add_argument("dataset", type=str, choices=["something", "jester", "moments"])
parser.add_argument("modality", type=str, choices=["RGB", "Flow"])
parser.add_argument("--train-list", type=Path, required=True)
parser.add_argument("--val-list", type=Path, required=True)
parser.add_argument("--categories", type=Path, required=True)
parser.add_argument("--root-path", type=Path, required=True)
parser.add_argument(
    "--image-prefix",
    type=str,
    default="{:05d}.jpg",
    help="Python formatting string for JPEG files",
)
parser.add_argument("--experiment-dir", default=Path(os.getcwd()), type=Path)
# ========================= Model Configs ==========================
parser.add_argument("--arch", type=str, default="BNInception")
parser.add_argument("--segment-count", type=int, default=3)
parser.add_argument(
    "--consensus-type",
    type=str,
    choices=["avg", "TRN", "TRNmultiscale"],
    default="TRN",
    help="Method to fuse segment-wise predictions",
)

parser.add_argument(
    "--dropout",
    "--do",
    default=0.8,
    type=float,
    metavar="DO",
    help="dropout ratio",
)
parser.add_argument("--loss-type", type=str, default="nll", choices=["nll"])
parser.add_argument(
    "--img-feature-dim",
    default=256,
    type=int,
    help="the feature dimension for each frame",
)

# ========================= Learning Configs ==========================
parser.add_argument(
    "--epochs", default=120, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--lr-steps",
    default=[50, 100],
    type=float,
    nargs="+",
    metavar="LRSteps",
    help="epochs to decay learning rate by 10",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=5e-4,
    type=float,
    metavar="W",
    help="weight decay)",
)
parser.add_argument(
    "--clip-gradient",
    "--gd",
    default=20,
    type=float,
    metavar="W",
    help="gradient norm clipping",
)
parser.add_argument("--no-partialbn", "--npb", default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument(
    "--print-freq",
    "-p",
    default=20,
    type=int,
    metavar="N",
    help="print frequency",
)
parser.add_argument(
    "--eval-freq",
    "-ef",
    default=5,
    type=int,
    metavar="N",
    help="evaluation frequency",
)


# ========================= Runtime Configs ==========================
parser.add_argument(
    "-j",
    "--workers",
    default=cpu_count(),
    type=int,
    metavar="N",
    help="number of data loading workers",
)
parser.add_argument(
    "--resume",
    default=None,
    type=Path,
    metavar="PATH",
    help="path to latest checkpoint)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("--copy-local", action="store_true",
                    help="Copy dataset to --local-dir that specified by")

try:
    storage_loc = '/raid/local_scratch/' + os.environ['SLURM_JOB_USER']
except KeyError:
    storage_loc = Path('/tmp')
parser.add_argument("--local-dir", type=Path, default=storage_loc,
                    help="Path to fast local storage to store dataset")
parser.add_argument("--flow-prefix", default="", type=str)
parser.add_argument(
    "-v",
    "--verbose",
    dest="verbosity",
    action="count",
    default=0,
    help="Verbosity (between 1-4 occurrences with more leading to more "
    "verbose logging). CRITICAL=0, ERROR=1, WARN=2, INFO=3, "
    "DEBUG=4",
)
parser.add_argument("--progress-bar", action="store_true", help="Show progress bar")
parser.add_argument("--tensboard-log-freq", default=10, type=int,
                    help="Number of iterations between writing metrics to tensboard "
                         "summary")

# ========================= Runtime Configs ==========================
log_levels = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.WARN,
    3: logging.INFO,
    4: logging.DEBUG,
}
