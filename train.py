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
    return tokenizer(sequence["text"], truncation=True, max_length=128)

def encode_labels(example):
    """Map string labels to integers."""
    example["labels"] = label2id[example["labels"]]
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

def main(model_ckpt):
    # Setup
    global tokenizer, label2id

    gdrive_dir = Path('langid')
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Load and split dataset
    dataset = load_dataset("papluca/language-identification")
    ds_train = dataset['train']
    ds_valid = dataset['validation']
    ds_test = dataset['test']
    print(f"Train / valid / test samples: {len(ds_train)} / {len(ds_valid)} / {len(ds_test)}")

    # Prepare label mappings
    amazon_languages = ['en', 'de', 'fr', 'es', 'ja', 'zh']
    xnli_languages = ['ar', 'el', 'hi', 'ru', 'th', 'tr', 'vi', 'bg', 'sw', 'ur']
    stsb_languages = ['it', 'nl', 'pl', 'pt']
    all_langs = sorted(list(set(amazon_languages + xnli_languages + stsb_languages)))
    id2label = {idx: all_langs[idx] for idx in range(len(all_langs))}
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
    output_dir = gdrive_dir / f"{model_ckpt.split('/')[-1]}-finetuned-language-detection"

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
    model_preds = [s['label'] for s in pipe(ds_test.text.values.tolist(), truncation=True, max_length=128)]
    print(f"Our model time: {time.perf_counter() - start_time:.2f} seconds")
    print("Our Model Classification Report:")
    print(classification_report(ds_test.labels.values.tolist(), model_preds, digits=3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language identification model")
    parser.add_argument("--model_checkpoint", type=str, default="xlm-roberta-base",
                        help="Model checkpoint to use for training")
    args = parser.parse_args()
    
    main(args.model_checkpoint)