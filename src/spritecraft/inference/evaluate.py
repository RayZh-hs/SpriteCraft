"""Validation-set evaluation helpers."""

from pathlib import Path

from spritecraft.config import CHECKPOINTS_DIR, IMAGE_SIZE, OUTPUT_DIR
from spritecraft.data.dataset import TextureDataset
from spritecraft.inference.sampler import load_model, sample_tokens, save_prediction_bundle


def _resolve_index(dataset: TextureDataset, index: int, filename: str | None) -> int:
    if filename is not None:
        try:
            return dataset.filenames.index(filename)
        except ValueError as exc:
            raise ValueError(f"{filename!r} is not present in the {dataset.split} split") from exc

    if index < 0 or index >= len(dataset):
        raise IndexError(f"index {index} is out of range for split {dataset.split!r} (size={len(dataset)})")
    return index


def run(
    checkpoint_dir: str | Path = CHECKPOINTS_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    split: str = "val",
    index: int = 0,
    filename: str | None = None,
):
    """Evaluate one dataset example and write a full comparison bundle."""
    dataset = TextureDataset(split=split)
    if len(dataset) == 0:
        raise ValueError(f"Dataset split {split!r} is empty. Run preprocessing first.")

    dataset_index = _resolve_index(dataset, index=index, filename=filename)
    sample = dataset[dataset_index]

    model, checkpoint_path = load_model(checkpoint_dir)
    prediction = sample_tokens(
        model,
        sample["content_ref"].unsqueeze(0),
        sample["style_ref"].unsqueeze(0),
    ).squeeze(0)

    bundle_name = (
        f"{split}_{dataset_index:03d}_{Path(sample['filename']).stem}"
        f"_styled_like_{Path(sample['style_filename']).stem}"
    )
    result = save_prediction_bundle(
        output_dir=output_dir,
        bundle_name=bundle_name,
        content_tokens=sample["content_ref"],
        content_size=IMAGE_SIZE,
        style_tokens=sample["style_ref"],
        style_size=IMAGE_SIZE,
        prediction_tokens=prediction,
        prediction_size=IMAGE_SIZE,
        truth_tokens=sample["target"],
        truth_size=IMAGE_SIZE,
        metadata={
            "split": split,
            "index": dataset_index,
            "filename": sample["filename"],
            "content_filename": sample["content_filename"],
            "style_filename": sample["style_filename"],
            "target_filename": sample["target_filename"],
            "content_pack": sample["content_pack"],
            "style_pack": sample["style_pack"],
            "target_pack": sample["target_pack"],
            "checkpoint_path": str(checkpoint_path.resolve()),
        },
    )

    print(f"Evaluated split={split} index={dataset_index} filename={sample['filename']}")
    print(f"Style reference: {sample['style_filename']} from pack {sample['style_pack']}")
    print(f"Saved original texture to {result['original_path']}")
    print(f"Saved style reference to {result['style_path']}")
    print(f"Saved generated texture to {result['produced_path']}")
    print(f"Saved source of truth to {result['truth_path']}")
    print(f"Saved side-by-side comparison to {result['comparison_path']}")

    metrics = result["metrics"]
    if metrics is not None:
        print(
            "Metrics: "
            f"pixel_accuracy={metrics['pixel_accuracy']:.4f} "
            f"rgb_mae={metrics['rgb_mae']:.2f} "
            f"exact_match={metrics['exact_match']}"
        )

    return result["bundle_dir"]
