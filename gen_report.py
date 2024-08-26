import argparse
import json
from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_dataset, IterableDataset, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sc_types import ComparisonItem, GroupedComparison, ModelEvaluation


def load_model_and_tokenizer(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded successfully")
    return model, tokenizer


def gen_report(
    model_path: str, dataset_name: str, split: str, text_column: str, label_column: str
):
    # Load the saved model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Create a pipeline for text classification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, device=device
    )

    # Load the specified dataset
    dataset = load_dataset(dataset_name)

    if isinstance(dataset, IterableDataset):
        raise ValueError(
            f"Dataset {dataset_name} is an IterableDataset. Please use a Dataset instead."
        )

    ds_eval = dataset[split]

    if not isinstance(ds_eval, Dataset):
        raise ValueError(
            f"Split {split} is not a Dataset. Currently {type(ds_eval)} is not supported."
        )

    # Get predictions
    total_samples = len(ds_eval[text_column])
    batch_size = 128 if torch.cuda.is_available() else 32
    predictions = []
    print(f"Starting inference on {total_samples} samples...")
    for i in tqdm(range(0, total_samples, batch_size), desc="Inference Progress"):
        batch = ds_eval[text_column][i : i + batch_size]
        batch_predictions = classifier(batch, truncation=True, max_length=128)
        predictions.extend(batch_predictions)

    predicted_labels = [pred["label"] for pred in predictions]
    true_labels = ds_eval[label_column]

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Create wrong_predictions dictionary
    wrong_predictions: Dict[str, GroupedComparison] = {}
    for true_label, pred_label, text in zip(
        true_labels, predicted_labels, ds_eval[text_column]
    ):
        true_label, pred_label = str(true_label), str(pred_label)
        if true_label != pred_label:
            if true_label not in wrong_predictions:
                wrong_predictions[true_label] = GroupedComparison(model_preds={})
            if pred_label not in wrong_predictions[true_label].model_preds:
                wrong_predictions[true_label].model_preds[pred_label] = []

            wrong_predictions[true_label].model_preds[pred_label].append(
                ComparisonItem(
                    path=None,
                    data=text,
                    true_label=str(true_label),
                    model_pred=str(pred_label),
                )
            )

    # Create ModelEvaluation object
    evaluation = ModelEvaluation(
        model_name=model_path,
        accuracy=float(accuracy),
        wrong_predictions=wrong_predictions,
    )

    output_path = model_path.replace("/", "_")
    output_path = f"{output_path}_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation.dict(), f, ensure_ascii=False, indent=4)

    # Print evaluation results
    print(f"Model: {evaluation.model_name}")
    print(f"Accuracy: {evaluation.accuracy:.4f}")
    print("\nWrong Predictions Summary:")
    for true_label, grouped_comparison in evaluation.wrong_predictions.items():
        print(f"\nTrue Label: {true_label}")
        for pred_label, items in grouped_comparison.model_preds.items():
            print(f"  Predicted as {pred_label}: {len(items)} times")
            # Print a few examples
            for item in items[:3]:
                print(f"    {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a text classification model with structured output"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to evaluate"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the Hugging Face dataset"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
        help="Name of the column containing the text data",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        required=True,
        help="Name of the column containing the labels",
    )
    args = parser.parse_args()

    gen_report(
        args.model_path, args.dataset, args.split, args.text_column, args.label_column
    )
