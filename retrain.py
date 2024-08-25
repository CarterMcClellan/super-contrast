import time
from pathlib import Path
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

# Helper functions (unchanged)
def tokenize_text(sequence):
    return tokenizer(sequence["text"], truncation=True, max_length=128)

def encode_labels(example):
    example["labels"] = label2id[example["labels"]]
    return example

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1
    }

def main(model_path):
    global tokenizer, label2id

    gdrive_dir = Path('langid')
    
    # Load the existing model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Load the original dataset
    original_dataset = load_dataset("CarterMcClellan/cmac_adv_langid")
    
    # Combine datasets (you can adjust this based on your needs)
    combined_train = original_dataset['train']
    combined_valid = original_dataset['validation']
    ds_test = original_dataset['test']

    print(f"Combined train / valid / test samples: {len(combined_train)} / {len(combined_valid)} / {len(ds_test)}")

    # Prepare label mappings (assuming the same labels as before)
    amazon_languages = ['en', 'de', 'fr', 'es', 'ja', 'zh']
    xnli_languages = ['ar', 'el', 'hi', 'ru', 'th', 'tr', 'vi', 'bg', 'sw', 'ur']
    stsb_languages = ['it', 'nl', 'pl', 'pt']
    all_langs = sorted(list(set(amazon_languages + xnli_languages + stsb_languages)))
    id2label = {idx: all_langs[idx] for idx in range(len(all_langs))}
    label2id = {v: k for k, v in id2label.items()}

    # Tokenize and encode datasets
    tok_train = combined_train.map(tokenize_text, batched=True).map(encode_labels, batched=False)
    tok_valid = combined_valid.map(tokenize_text, batched=True).map(encode_labels, batched=False)
    tok_test = ds_test.map(tokenize_text, batched=True).map(encode_labels, batched=False)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training setup (adjusted for fine-tuning)
    epochs = 1  # Reduced number of epochs
    lr = 1e-5  # Lower learning rate
    train_bs = 32  # Smaller batch size
    eval_bs = train_bs * 2
    logging_steps = len(tok_train) // train_bs
    output_dir = gdrive_dir / f"{model_path.split('/')[-1]}-further-finetuned-language-detection"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=logging_steps,
        fp16=True,  # Remove if GPU doesn't support it
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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

    # Fine-tune the model
    trainer.train()

    # Benchmarking
    ds_test = ds_test.to_pandas()

    # Our model benchmark
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    start_time = time.perf_counter()
    model_preds = [s['label'] for s in pipe(ds_test.text.values.tolist(), truncation=True, max_length=128)]
    print(f"Fine-tuned model time: {time.perf_counter() - start_time:.2f} seconds")
    print("Fine-tuned Model Classification Report:")
    print(classification_report(ds_test.labels.values.tolist(), model_preds, digits=3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language identification model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument("--new_data_path", type=str, required=True,
                        help="Path to the new data for fine-tuning")
    args = parser.parse_args()
    
    main(args.model_path, args.new_data_path)