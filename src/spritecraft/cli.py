"""Command-line interface."""

import argparse

from spritecraft.config import MIN_SHARED_PACKS


def main():
    parser = argparse.ArgumentParser(description="SpriteCraft")
    subparsers = parser.add_subparsers(dest="command")

    preprocess_parser = subparsers.add_parser("preprocess", help="Run data preprocessing")
    preprocess_parser.add_argument("--packs-dir", type=str, default="data/raw_packs")
    preprocess_parser.add_argument("--manifest", type=str, default=None)
    preprocess_parser.add_argument("--min-shared-packs", type=int, default=MIN_SHARED_PACKS)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    train_parser.add_argument("--steps", type=int, default=100_000)

    generate_parser = subparsers.add_parser("generate", help="Generate textures from a resource pack")
    generate_parser.add_argument("--pack", type=str, required=True)
    generate_parser.add_argument("--checkpoint", type=str, default="checkpoints")
    generate_parser.add_argument("--output", type=str, default="output")
    select_group = generate_parser.add_mutually_exclusive_group(required=True)
    select_group.add_argument("--textures", nargs="+", default=None)
    select_group.add_argument("--random", type=int, dest="random_count")
    generate_parser.add_argument("--seed", type=int, default=None)

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
        train.run(args.checkpoint_dir, args.steps)
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
    else:
        parser.print_help()
