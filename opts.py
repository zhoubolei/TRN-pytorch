import configargparse
from multiprocessing import cpu_count

parser = configargparse.ArgumentParser(
    description="PyTorch implementation of Temporal Segment Networks",
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-c', '--config-file', is_config_file=True,
                    help="Config file path")

parser.add_argument("dataset", type=str, choices=["something", "jester", "moments"])
parser.add_argument("modality", type=str, choices=["RGB", "Flow"])
parser.add_argument("--train-list", type=str, required=True)
parser.add_argument("--val-list", type=str, required=True)
parser.add_argument("--categories", type=str, required=True)
parser.add_argument("--root-path", type=str, required=True)
parser.add_argument("--image-prefix", type=str, default="{:05d}.jpg",
                    help="Python formatting string for JPEG files")
parser.add_argument("--store-name", type=str)
# ========================= Model Configs ==========================
parser.add_argument("--arch", type=str, default="BNInception")
parser.add_argument("--num-segments", type=int, default=3)
parser.add_argument("--consensus-type", type=str, default="avg")

parser.add_argument(
    "--dropout",
    "--do",
    default=0.8,
    type=float,
    metavar="DO",
    help="dropout ratio (default: 0.5)",
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
    help="mini-batch size (default: 256)",
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
    help="weight decay (default: 5e-4)",
)
parser.add_argument(
    "--clip-gradient",
    "--gd",
    default=20,
    type=float,
    metavar="W",
    help="gradient norm clipping (default: disabled)",
)
parser.add_argument("--no-partialbn", "--npb", default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument(
    "--print-freq",
    "-p",
    default=20,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--eval-freq",
    "-ef",
    default=5,
    type=int,
    metavar="N",
    help="evaluation frequency (default: 5)",
)


# ========================= Runtime Configs ==========================
parser.add_argument(
    "-j",
    "--workers",
    default=cpu_count(),
    type=int,
    metavar="N",
    help="number of data loading workers (default: n_cpus)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument("--snapshot-pref", type=str, default="")
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("--flow-prefix", default="", type=str)
parser.add_argument("--root-log", type=str, default="log")
parser.add_argument("--root-model", type=str, default="model")
parser.add_argument("--root-output", type=str, default="output")
