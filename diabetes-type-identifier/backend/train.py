"""
train.py - Standalone training script
Usage: python train.py [--data path/to/data.csv] [--model random_forest|decision_tree]
"""

import argparse
import os
import sys

# Allow imports from backend/
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import load_dataset
from model import train_model


def main():
    parser = argparse.ArgumentParser(description="Train the diabetes classifier")
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "sample_diabetes.csv"),
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "decision_tree"],
        help="Model type to train"
    )
    args = parser.parse_args()

    print(f"Loading dataset from: {args.data}")
    df = load_dataset(args.data)
    print(f"Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")

    print(f"Training {args.model} model...")
    metrics = train_model(df, model_type=args.model)

    print("\n=== Training Complete ===")
    print(f"Model type     : {metrics['model_type']}")
    print(f"Training samples (after SMOTE): {metrics['training_samples']}")
    print(f"Weighted F1    : {metrics['f1_score']}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print("\nClassification Report:")
    for label, scores in metrics["classification_report"].items():
        if isinstance(scores, dict):
            print(f"  {label}: precision={scores['precision']:.3f}, "
                  f"recall={scores['recall']:.3f}, f1={scores['f1-score']:.3f}")

    print("\nModel saved to ../models/trained_model.pkl")


if __name__ == "__main__":
    main()
