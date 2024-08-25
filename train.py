import time
from pathlib import Path
from statistics import mean, stdev
import argparse

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    pipeline,
    Trainer,
    TrainingArguments
)

# Helper functions
def tokenize_text(sequence):
    """Tokenize input sequence."""
    return tokenizer(sequence[text_column], truncation=True, max_length=128)

def encode_labels(example):
    """Map string labels to integers."""
    example["labels"] = label2id[example[label_column]]
    return example

def compute_metrics(pred):
    """Custom metric to be used during training."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)  # Accuracy
    f1 = f1_score(labels, preds, average="weighted")  # F1-score
    return {
        "accuracy": acc,
        "f1": f1
    }

def train(model_ckpt, dataset_name, text_col, label_col):
    # Setup
    global tokenizer, label2id, text_column, label_column
    text_column = text_col
    label_column = label_col

    normalized_name = dataset_name.replace('/', '_')
    gdrive_dir = Path(f'{normalized_name}_model_finetuning')
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Load and split dataset
    dataset = load_dataset(dataset_name)
    if 'validation' not in dataset.keys():
        print("No validation set found. Creating validation set from train set...")
        split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
        ds_train = split_dataset['train']
        ds_valid = split_dataset['test']
    else:
        ds_train = dataset['train']
        ds_valid = dataset['validation']

    ds_test = dataset['test']
    print(f"Train / valid / test samples: {len(ds_train)} / {len(ds_valid)} / {len(ds_test)}")

    # Prepare label mappings
    all_langs = sorted(list(set(ds_train[label_column])))
    id2label = {idx: lang for idx, lang in enumerate(all_langs)}
    label2id = {v: k for k, v in id2label.items()}

    # Tokenize and encode datasets
    tok_train = ds_train.map(tokenize_text, batched=True).map(encode_labels, batched=False)
    tok_valid = ds_valid.map(tokenize_text, batched=True).map(encode_labels, batched=False)
    tok_test = ds_test.map(tokenize_text, batched=True).map(encode_labels, batched=False)

    # Corpus statistics
    _len = [len(sample) for sample in tok_train['input_ids']]
    avg_len, std_len = mean(_len), stdev(_len)
    min_len, max_len = min(_len), max(_len)
    print('-'*10 + ' Corpus statistics ' + '-'*10)
    print(f'\nAvg. length: {avg_len:.1f} (std. {std_len:.1f})')
    print('Min. length:', min_len)
    print('Max. length:', max_len)

    # Model preparation
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=len(all_langs), id2label=id2label, label2id=label2id
    )

    # Training setup
    epochs = 2
    lr = 2e-5
    train_bs = 64
    eval_bs = train_bs * 2
    logging_steps = len(tok_train) // train_bs
    output_dir = gdrive_dir / f"{model_ckpt.split('/')[-1]}-finetuned"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        evaluation_strategy="epoch",
        logging_steps=logging_steps,
        fp16=True,  # Remove if GPU doesn't support it
    )

    trainer = Trainer(
        model,
        training_args,
        compute_metrics=compute_metrics,
        train_dataset=tok_train,
        eval_dataset=tok_valid,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Benchmarking
    ds_test = ds_test.to_pandas()

    # Our model benchmark
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    start_time = time.perf_counter()
    model_preds = [s['label'] for s in pipe(ds_test[text_column].values.tolist(), truncation=True, max_length=128)]
    print(f"Our model time: {time.perf_counter() - start_time:.2f} seconds")
    print("Our Model Classification Report:")
    print(classification_report(ds_test[label_column].values.tolist(), model_preds, digits=3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language identification model")
    parser.add_argument("--model_checkpoint", type=str, default="xlm-roberta-base",
                        help="Model checkpoint to use for training")
    parser.add_argument("--dataset", type=str, default="papluca/language-identification",
                        help="Dataset to use for training")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Name of the column containing the text data")
    parser.add_argument("--label_column", type=str, default="labels",
                        help="Name of the column containing the labels")
    args = parser.parse_args()
    
    train(args.model_checkpoint, args.dataset, args.text_column, args.label_column)