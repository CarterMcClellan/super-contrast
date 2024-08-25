import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Union

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

# Custom types
from sc_types import (
    AdversarialExample,
    GroupedComparison,
    ModelEvaluation,
)
from sc_text import TaskGenerator, AdversarialGenerator, AdversarialItem, AdversarialResponse

def read_report(report_path: str) -> ModelEvaluation:
    with open(report_path, "r") as f:
        report = json.load(f)
    return ModelEvaluation(**report)

def analyze_misclassifications(
    grouped_comparisons: Dict[str, GroupedComparison]
) -> Dict[str, Dict[str, int]]:
    misclassifications = defaultdict(lambda: defaultdict(int))
    for true_label, group in grouped_comparisons.items():
        for pred_label, items in group.model_preds.items():
            misclassifications[true_label][pred_label] = len(items)
    return misclassifications # type: ignore

def extract_top_trends(
    misclassifications: Dict[str, Dict[str, int]], top_n: int = 5
) -> List[Tuple[str, str, int]]:
    trends = []
    for true_label, predictions in misclassifications.items():
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        for pred_label, count in sorted_preds[:top_n]:
            trends.append((true_label, pred_label, count))
    return sorted(trends, key=lambda x: x[2], reverse=True)

def create_adversarial_tasks(
    data: Dict[str, GroupedComparison], output_dir: str, modality: str
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    tasks = []

    task_generator = None
    if modality == "text":
        task_generator = TaskGenerator()
    elif modality == "image":
        raise NotImplementedError("Image modality is not yet supported")
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    for true_label, group in data.items():
        for pred_label, examples in group.model_preds.items():
            if modality == "text":
                example_data = [item.data for item in examples]

                if len(example_data) <= 5:
                    print(f"Skipping task generation for {true_label} -> {pred_label}")
                    print("Not enough examples to generate a task")
                    continue

                task = task_generator.generate(
                    pred_label=pred_label,
                    true_label=true_label,
                    examples=example_data, # type: ignore
                )

                adv_prompt = AdversarialExample(
                    examples=example_data, # type: ignore
                    predicted_label=pred_label,
                    true_label=true_label,
                    task=task.task,
                )

                tasks.append(adv_prompt.dict())

                with open(
                    os.path.join(
                        output_dir, f"adv_prompt_{true_label}_{pred_label}.json"
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(adv_prompt.dict(), f, ensure_ascii=False, indent=4)

            elif modality == "image":
                raise NotImplementedError("Image modality is not yet supported")
            else:
                raise ValueError(f"Unsupported modality: {modality}")
    
    return tasks

def analyze_trends(data: ModelEvaluation):
    print(f"Analyzing trends in misclassifications for {data.model_name}...")

    trends = extract_top_trends(
        analyze_misclassifications(data.wrong_predictions)
    )
    for true_label, pred_label, count in trends[:5]:
        print(f"  True label {true_label} misclassified as {pred_label}: {count} times")
    
    print(f"\nAccuracy for {data.model_name}: {data.accuracy:.2%}")
    print(f"Total wrong predictions: {sum(len(items) for group in data.wrong_predictions.values() for items in group.model_preds.values())}")

def generate_new_examples(task: Dict, generator: AdversarialGenerator) -> List[AdversarialItem]:
    response: AdversarialResponse = generator.generate(
        task=task['task'],
        examples=task['examples']
    )
    return response.data

def save_flattened_examples(all_examples: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    filename = "flattened_adversarial_examples.json"
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=4)

def load_adversarial_examples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def combine_datasets(original_dataset, adversarial_examples, text_column, label_column):
    # Ensure the adversarial examples have the same structure as the original dataset
    adversarial_data = {
        text_column: [example[text_column] for example in adversarial_examples],
        label_column: [example[label_column] for example in adversarial_examples]
    }
    
    # Create a new dataset from the adversarial examples
    adversarial_dataset = Dataset.from_dict(adversarial_data)
    
    # Combine the original dataset with the adversarial examples
    combined_train = concatenate_datasets([original_dataset['train'], adversarial_dataset])
    
    # Create a new DatasetDict with the combined training set
    return DatasetDict({
        'train': combined_train,
        'validation': original_dataset['validation'],
        'test': original_dataset['test']
    })

def upload_combined_dataset(output_dir: str, dataset_name: str, original_dataset_name: str, text_column: str, label_column: str):
    # Load the original dataset
    original_dataset = load_dataset(original_dataset_name)
    
    # Load your new adversarial examples
    adversarial_examples = load_adversarial_examples(os.path.join(output_dir, "flattened_adversarial_examples.json"))
    
    # Combine datasets
    combined_dataset = combine_datasets(original_dataset, adversarial_examples, text_column, label_column)
    
    # Push to Hugging Face Hub
    combined_dataset.push_to_hub(dataset_name)
    print(f"Uploaded combined dataset to {dataset_name}")

def gen_adversarial(report_path: str, dataset_name: str, original_dataset_name: str, text_column: str, label_column: str):
    # Part 1: Analyze model evaluation results
    report = read_report(report_path)
    analyze_trends(report)

    # Part 2: Create adversarial tasks
    tasks_dir = f"adversarial_tasks/{report.model_name}_wrong"
    tasks = create_adversarial_tasks(
        report.wrong_predictions,
        tasks_dir,
        "text",  # Assuming text modality for model evaluation
    )

    # Part 3: Generate new adversarial examples
    output_dir = "new_training_examples"
    generator = AdversarialGenerator()
    all_examples = []

    for task in tasks:
        try:
            new_examples = generate_new_examples(task, generator)
        except:
            print(f"Failed to generate new examples for {task['true_label']} (predicted as {task['predicted_label']})")
            continue
        for example in new_examples:
            flattened_example = {
                text_column: example.text,
                label_column: task['true_label']
            }
            all_examples.append(flattened_example)
        
        print(f"Generated {len(new_examples)} new examples for {task['true_label']} (predicted as {task['predicted_label']})")
    
    save_flattened_examples(all_examples, output_dir)
    print(f"Saved {len(all_examples)} flattened examples to {output_dir}/flattened_adversarial_examples.json")
    upload_combined_dataset(output_dir, dataset_name, original_dataset_name, text_column, label_column)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze model evaluation results, generate adversarial examples, and upload combined dataset"
    )
    parser.add_argument(
        "--report_path",
        type=str,
        required=True,
        help="Path to the JSON file containing model evaluation results",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name for the new dataset on Hugging Face Hub (e.g., 'your-username/dataset-adversarial')",
    )
    parser.add_argument(
        "--original_dataset_name",
        type=str,
        required=True,
        help="Name of the original dataset on Hugging Face Hub",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
        help="Name of the column containing the text data in the dataset",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        required=True,
        help="Name of the column containing the labels in the dataset",
    )
    args = parser.parse_args()

    gen_adversarial(args.report_path, args.dataset_name, args.original_dataset_name, args.text_column, args.label_column)