"""Command-line interface."""

import argparse

from spritecraft.config import MIN_SHARED_PACKS


def main():
    parser = argparse.ArgumentParser(description="SpriteCraft")
    subparsers = parser.add_subparsers(dest="command")

    def add_debug_parser(name: str):
        debug_parser = subparsers.add_parser(name, help="Inspect or control a running trainer")
        debug_parser.add_argument(
            "debug_action",
            nargs="?",
            choices=("status", "snapshot", "preview"),
            default="status",
        )
        debug_parser.add_argument("--pid", type=int, default=None)
        debug_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
        debug_parser.add_argument("--wait-timeout", type=float, default=600.0)
        return debug_parser

    preprocess_parser = subparsers.add_parser("preprocess", help="Run data preprocessing")
    preprocess_parser.add_argument("--packs-dir", type=str, default="data/raw_packs")
    preprocess_parser.add_argument("--manifest", type=str, default=None)
    preprocess_parser.add_argument("--min-shared-packs", type=int, default=MIN_SHARED_PACKS)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    train_parser.add_argument("--steps", type=int, default=10_000)
    train_parser.add_argument("--pack", type=str, default=None, help="Train specific pack (default: all)")

    generate_parser = subparsers.add_parser("generate", help="Generate textures from a resource pack")
    generate_parser.add_argument("--pack", type=str, required=True)
    generate_parser.add_argument("--checkpoint", type=str, default="checkpoints")
    generate_parser.add_argument("--output", type=str, default="output")
    select_group = generate_parser.add_mutually_exclusive_group(required=True)
    select_group.add_argument("--textures", nargs="+", default=None)
    select_group.add_argument("--random", type=int, dest="random_count")
    generate_parser.add_argument("--seed", type=int, default=None)

    add_debug_parser("debug")

    args = parser.parse_args()

    if args.command == "preprocess":
        from spritecraft.data import preprocess
        preprocess.run(
            packs_dir=args.packs_dir,
            manifest_path=args.manifest,
            min_shared_packs=args.min_shared_packs,
        )
    elif args.command == "train":
        from spritecraft.training import train
        train.run(args.checkpoint_dir, args.steps, args.pack)
    elif args.command == "generate":
        from spritecraft.inference import generate
        generate.run(
            pack_id=args.pack,
            checkpoint=args.checkpoint,
            output_dir=args.output,
            textures=args.textures,
            random_count=args.random_count,
            seed=args.seed,
        )
    elif args.command == "debug":
        from spritecraft.debug import debug
        debug.run(
            action=args.debug_action,
            pid=args.pid,
            checkpoint_dir=args.checkpoint_dir,
            wait_timeout=args.wait_timeout,
        )
    else:
        parser.print_help()
