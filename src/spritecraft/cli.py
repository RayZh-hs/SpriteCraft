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

    sample_parser = subparsers.add_parser("sample", help="Generate textures")
    sample_parser.add_argument("--content", type=str, required=True)
    sample_parser.add_argument("--support-original", action="append", dest="support_originals", required=True)
    sample_parser.add_argument("--support-styled", action="append", dest="support_styleds", required=True)
    sample_parser.add_argument("--truth", type=str, default=None)
    sample_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    sample_parser.add_argument("--output", type=str, default="output")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the validation matrix or one dataset example")
    evaluate_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    evaluate_parser.add_argument("--output", type=str, default="output")
    evaluate_parser.add_argument("--split", type=str, choices=("train", "val"), default="val")
    evaluate_parser.add_argument("--mode", type=str, choices=("matrix", "single"), default="matrix")
    evaluate_parser.add_argument("--index", type=int, default=0)
    evaluate_parser.add_argument("--filename", type=str, default=None)
    evaluate_parser.add_argument("--target-pack", type=str, default=None)

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
    elif args.command == "sample":
        from spritecraft.inference import sampler
        sampler.run(
            content_path=args.content,
            support_original_paths=args.support_originals,
            support_styled_paths=args.support_styleds,
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
            mode=args.mode,
            index=args.index,
            filename=args.filename,
            target_pack=args.target_pack,
        )
    else:
        parser.print_help()
