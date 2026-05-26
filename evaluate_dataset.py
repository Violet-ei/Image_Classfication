"""Command-line evaluation for a class-folder dataset zip.

Example:
    python evaluate_dataset.py --dataset dataset.zip --models all --topk 5 --batch-size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from classifier import FourMethodClassifier, MODEL_SPECS, analyze_disagreements
from data_utils import discover_class_folder_dataset, safe_extract_zip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset zip or extracted dataset root")
    parser.add_argument("--models", default="all", help="Comma-separated model keys or 'all'")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--topn-disagreement", type=int, default=8)
    parser.add_argument("--out-dir", default="eval_outputs")
    parser.add_argument("--device", default=None, help="cpu/cuda/mps; default auto")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    if dataset_path.suffix.lower() == ".zip":
        dataset_root = safe_extract_zip(dataset_path, out_dir / "extracted_dataset")
    else:
        dataset_root = dataset_path

    samples, dataset_labels = discover_class_folder_dataset(dataset_root)
    if args.models == "all":
        keys = list(MODEL_SPECS.keys())
    else:
        keys = [x.strip() for x in args.models.split(",") if x.strip()]

    clf = FourMethodClassifier(device=args.device)
    metrics_df, pred_df = clf.evaluate_dataset(samples, dataset_labels, keys, topk=args.topk, batch_size=args.batch_size)
    dis_df = analyze_disagreements(pred_df, topn=args.topn_disagreement)

    metrics_path = out_dir / "model_accuracy_summary.csv"
    pred_path = out_dir / "all_model_predictions.csv"
    dis_path = out_dir / "largest_disagreement_samples.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    dis_df.to_csv(dis_path, index=False, encoding="utf-8-sig")

    print("Dataset root:", dataset_root)
    print("Labels:", ", ".join(dataset_labels))
    print("Samples:", len(samples))
    print("\nAccuracy summary:")
    print(metrics_df.to_string(index=False))
    print("\nLargest disagreements:")
    print(dis_df[["file", "true_label", "vote_distribution", "correct_model_count", "analysis_reason"]].to_string(index=False))
    print("\nSaved:")
    print(metrics_path)
    print(pred_path)
    print(dis_path)


if __name__ == "__main__":
    main()
