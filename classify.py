"""Troca da LM Head por uma cabeça de classificação binária (SST-2).

Compara fine-tuning completo x transfer learning (backbone congelado)
com qualquer backbone do Hugging Face (BERT, DistilBERT, Llama, ...).

Exemplos:
    python classify.py --model distilbert-base-uncased
    python classify.py --model distilbert-base-uncased --freeze-backbone
    python classify.py --model meta-llama/Llama-3.2-1B   # requer HF_TOKEN
"""
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="treina somente a nova camada de classificação")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss_sum += out.loss.item()
            correct += (out.logits.argmax(-1) == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return loss_sum / len(loader), correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    if tokenizer.pad_token is None:  # Llama não tem pad token
        tokenizer.pad_token = tokenizer.eos_token

    # bfloat16 só em GPU compatível; senão float32
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, dtype=dtype, token=token
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.freeze_backbone:
        for param in model.base_model.parameters():
            param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros totais:     {total:,}")
    print(f"Parâmetros treináveis: {trainable:,}")

    ds = load_dataset("stanfordnlp/sst2")
    train = ds["train"].shuffle(seed=args.seed).select(range(args.train_size))
    val = ds["validation"]

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=args.max_length)

    cols = ["input_ids", "attention_mask", "labels"]
    train = train.map(tok, batched=True).rename_column("label", "labels")
    val = val.map(tok, batched=True).rename_column("label", "labels")
    train = train.select_columns(cols)
    val = val.select_columns(cols)

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=0.01,
    )

    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        val_loss, val_acc = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / len(train_loader),
               "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(row)
    elapsed = time.time() - start

    os.makedirs(args.out, exist_ok=True)
    mode = "head_only" if args.freeze_backbone else "full_finetune"
    name = f"{args.model.replace('/', '_')}__{mode}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump({"model": args.model, "mode": mode, "total_params": total,
                   "trainable_params": trainable, "train_seconds": elapsed,
                   "final_val_acc": history[-1]["val_acc"], "history": history}, f, indent=2)
    print(f"Tempo de treino: {elapsed:.1f}s | salvo em {args.out}/{name}")


if __name__ == "__main__":
    main()
