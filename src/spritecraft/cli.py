"""Command-line interface."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="SpriteCraft")
    subparsers = parser.add_subparsers(dest="command")

    preprocess_parser = subparsers.add_parser("preprocess", help="Run data preprocessing")
    preprocess_parser.add_argument("--packs-dir", type=str, default="data/raw_packs")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    train_parser.add_argument("--steps", type=int, default=100_000)

    sample_parser = subparsers.add_parser("sample", help="Generate textures")
    sample_parser.add_argument("--content", type=str, required=True)
    sample_parser.add_argument("--style", type=str, required=True)
    sample_parser.add_argument("--output", type=str, default="output")

    args = parser.parse_args()

    if args.command == "preprocess":
        from spritecraft.data import preprocess
        preprocess.run(args.packs_dir)
    elif args.command == "train":
        from spritecraft.training import train
        train.run(args.checkpoint_dir, args.steps)
    elif args.command == "sample":
        from spritecraft.inference import sampler
        sampler.run(args.content, args.style, args.output)
    else:
        parser.print_help()
