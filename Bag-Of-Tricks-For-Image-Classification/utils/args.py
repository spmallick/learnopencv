import argparse


def get_program_level_args():
    parser = argparse.ArgumentParser(description="Classification Training")

    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="Number of data loading workers",
    )

    parser.add_argument(
        "--use-smoothing",
        action="store_true",
        default=False,
        help="Use label smoothing trick",
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Coefficient for label smoothing (from 0.0 to 1.0 where 0.0 means no smoothing)",
    )

    parser.add_argument(
        "--use-mixup",
        action="store_true",
        default=False,
        help="Use mixup augmentation during training",
    )

    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="Alpha value for mixup augmentation",
    )

    parser.add_argument(
        "--use-cosine-scheduler",
        action="store_true",
        default=False,
        help="Use Cosine LR Scheduler instead of MultiStep",
    )

    parser.add_argument(
        "--use-knowledge-distillation",
        action="store_true",
        default=False,
        help="Use Knowledge Distillation technique",
    )

    parser.add_argument(
        "--distill-alpha", type=float, default=0.5, help="Distillation strength",
    )

    parser.add_argument(
        "--distill-temperature",
        type=int,
        default=20,
        help="Temperature hyper-parameter to make the outputs smoother for KD",
    )

    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="Evaluate model on validation set",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the trained model for evaluation",
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed to initialize all random generators",
    )

    return parser
