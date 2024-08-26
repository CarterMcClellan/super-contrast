import argparse
import json
import math
from typing import Dict, List, Union
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from PIL import Image
import open_clip

from sc_types import ComparisonItem, GroupedComparison, ModelEvaluation


def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    return model, preprocess, device


def encode_data(data: Union[str, Image.Image], clip_model, preprocess, device):
    if isinstance(data, str):
        text = open_clip.tokenize([data]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        return text_features.cpu().numpy()
    elif isinstance(data, Image.Image):
        image = preprocess(data).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        return image_features.cpu().numpy()
    else:
        raise ValueError("Unsupported data type")


def cluster_data(
    data: List[ComparisonItem], clip_model, preprocess, device
) -> List[List[ComparisonItem]]:
    encodings = [
        encode_data(item.data, clip_model, preprocess, device) for item in data
    ]
    encodings = torch.tensor(encodings).squeeze(1)

    n_clusters = max(2, int(math.sqrt(len(data))))
    print(f"Clustering data into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(encodings)

    clustered_data = [[] for _ in range(n_clusters)]
    for item, cluster in zip(data, clusters):
        clustered_data[cluster].append(item)

    return clustered_data


def apply_clip_clustering(input_path: str, output_path: str):
    # Load the existing ModelEvaluation from JSON
    with open(input_path, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    evaluation = ModelEvaluation(**evaluation_data)

    # Load CLIP model for clustering
    clip_model, preprocess, device = load_clip_model()
    print(f"Using device: {device}")

    cluster_evaluation = ModelEvaluation(
        model_name=evaluation.model_name,
        accuracy=evaluation.accuracy,
        wrong_predictions={},
    )

    # Cluster wrong predictions
    print("Clustering wrong predictions...")
    for true_label, grouped_comparison in tqdm(evaluation.wrong_predictions.items()):

        cluster_evaluation.wrong_predictions[true_label] = GroupedComparison(
            model_preds={}
        )

        for pred_label, item_groups in grouped_comparison.model_preds.items():
            clustered_items = cluster_data(item_groups, clip_model, preprocess, device)

            for i, cluster in enumerate(clustered_items):
                cluster_evaluation.wrong_predictions[true_label].model_preds[
                    f"cluster_{i}"
                ] = cluster

    # Save the clustered ModelEvaluation to a new JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cluster_evaluation.dict(), f, ensure_ascii=False, indent=4)

    print(f"Clustered report saved to {output_path}")

    # Print evaluation results
    print(f"\nModel: {cluster_evaluation.model_name}")
    print(f"Accuracy: {cluster_evaluation.accuracy:.4f}")
    print("\nWrong Predictions Summary (Clustered):")
    for true_label, grouped_comparison in cluster_evaluation.wrong_predictions.items():
        print(f"\nTrue Label: {true_label}")
        for cluster_name, cluster in grouped_comparison.model_preds.items():
            pred_label = cluster[0].model_pred
            print("\n  Cluster:", cluster_name, f"(Predicted label: {pred_label})")
            for item in cluster[:3]:
                print(f"      {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply CLIP clustering to a model evaluation report"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input ModelEvaluation JSON file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the clustered ModelEvaluation JSON file",
    )
    args = parser.parse_args()

    apply_clip_clustering(args.input_path, args.output_path)
