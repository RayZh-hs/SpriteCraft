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
    sample_parser.add_argument("--truth", type=str, default=None)
    sample_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    sample_parser.add_argument("--output", type=str, default="output")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one dataset example")
    evaluate_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    evaluate_parser.add_argument("--output", type=str, default="output")
    evaluate_parser.add_argument("--split", type=str, choices=("train", "val"), default="val")
    evaluate_parser.add_argument("--index", type=int, default=0)
    evaluate_parser.add_argument("--filename", type=str, default=None)

    args = parser.parse_args()

    if args.command == "preprocess":
        from spritecraft.data import preprocess
        preprocess.run(args.packs_dir)
    elif args.command == "train":
        from spritecraft.training import train
        train.run(args.checkpoint_dir, args.steps)
    elif args.command == "sample":
        from spritecraft.inference import sampler
        sampler.run(
            content_path=args.content,
            style_path=args.style,
            output_dir=args.output,
            checkpoint_dir=args.checkpoint_dir,
            truth_path=args.truth,
        )
    elif args.command == "evaluate":
        from spritecraft.inference import evaluate
        evaluate.run(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output,
            split=args.split,
            index=args.index,
            filename=args.filename,
        )
    else:
        parser.print_help()
