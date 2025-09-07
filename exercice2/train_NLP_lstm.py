# lstm_text_completion.py
# ------------------------------------------------------------
# End-to-end LSTM language model for next-sentence completion.
# - Downloads public-domain books from Project Gutenberg
# - Builds a word-level dataset
# - Trains a pure LSTM language model (no Transformers)
# - Generates next sentences to complete a given prompt
#
# Public-domain sources (Gutenberg IDs):
#   Alice in Wonderland         -> 11
#   The Adventures of Sherlock  -> 1661
#   Pride and Prejudice         -> 1342
#   The Time Machine            -> 35
#   Dracula                     -> 345
# ------------------------------------------------------------

import os
import re
import math
import json
import time
import argparse
import random
from collections import Counter
from typing import List, Tuple

import requests
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------
# 1) Reproducibility helpers
# -----------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# 2) Data download & cleaning
# -----------------------------
GUTENBERG_IDS = {
    "alice": 11,
    "sherlock": 1661,
    "pride": 1342,
    "time_machine": 35,
    "dracula": 345,
}

START_MARKER = re.compile(r"\*\*\* START OF.+\*\*\*")
END_MARKER   = re.compile(r"\*\*\* END OF.+\*\*\*")

def gutenberg_urls(gid: int) -> List[str]:
    """A couple of common URL patterns for redundancy."""
    return [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]

def download_book(gid: int, out_path: str, timeout: int = 20) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for url in gutenberg_urls(gid):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "lstm-demo/1.0"})
            if r.status_code == 200 and len(r.text) > 10000:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                return True
        except Exception:
            pass
    return False

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Gutenberg header/footer when markers exist; otherwise return as-is."""
    lines = text.splitlines()
    start_idx, end_idx = 0, len(lines)
    for i, ln in enumerate(lines):
        if START_MARKER.search(ln):
            start_idx = i + 1
            break
    for i in range(len(lines) - 1, -1, -1):
        if END_MARKER.search(lines[i]):
            end_idx = i
            break
    core = "\n".join(lines[start_idx:end_idx]).strip()
    return core if len(core) > 1000 else text

# -----------------------------
# 3) Tokenisation & vocab
# -----------------------------
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)  # words or punctuation

SPECIALS = {
    "PAD": "<pad>",
    "UNK": "<unk>",
    "BOS": "<bos>",
    "EOS": "<eos>",
}

def sentence_split(text: str) -> List[str]:
    # Split on sentence enders, keeping them (so generation can count sentences)
    parts = re.split(r"([.!?])", text)
    sents = []
    for i in range(0, len(parts)-1, 2):
        s = (parts[i].strip() + parts[i+1]).strip()
        if s:
            sents.append(s)
    # Add any tail without terminal punctuation
    if len(parts) % 2 == 1 and parts[-1].strip():
        sents.append(parts[-1].strip())
    return sents

def tokenize(text: str, lower: bool = True) -> List[str]:
    if lower:
        text = text.lower()
    return TOKEN_RE.findall(text)

def build_vocab(tokens: List[str], min_freq: int = 2) -> Tuple[dict, dict]:
    counter = Counter(tokens)
    itos = [SPECIALS["PAD"], SPECIALS["UNK"], SPECIALS["BOS"], SPECIALS["EOS"]]
    for tok, c in counter.most_common():
        if c >= min_freq and tok not in SPECIALS.values():
            itos.append(tok)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos

def numericalise(tokens: List[str], stoi: dict) -> List[int]:
    unk = stoi[SPECIALS["UNK"]]
    return [stoi.get(t, unk) for t in tokens]

# -----------------------------
# 4) Dataset for LM
# -----------------------------
class LMStreamDataset(Dataset):
    """
    Constructs (seq_len) -> predict next token.
    We use a continuous token stream with <eos> between sentences.
    """
    def __init__(self, ids: List[int], seq_len: int):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        # Last index where a full seq_len+1 target exists
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# -----------------------------
# 5) Model: Pure LSTM LM
# -----------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 256, hidden: int = 512, layers: int = 2,
                 dropout: float = 0.2, tie_weights: bool = True):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
        if tie_weights:
            if hidden != emb:
                # To tie weights, hidden must equal emb; if not, we’ll project.
                self.proj = nn.Linear(hidden, emb, bias=False)
                self.tied = True
            else:
                self.proj = None
                self.tied = True
        else:
            self.proj = None
            self.tied = False

        if self.tied and self.proj is None:
            self.fc.weight = self.emb.weight

    def forward(self, x, hidden=None):
        # x: (B, T)
        e = self.drop(self.emb(x))          # (B, T, E)
        out, hidden = self.lstm(e, hidden)  # (B, T, H)
        out = self.drop(out)
        if self.tied and self.proj is not None:
            out = self.proj(out)            # (B, T, E)
            logits = out @ self.emb.weight.T
        else:
            logits = self.fc(out)           # (B, T, V)
        return logits, hidden

# -----------------------------
# 6) Training & evaluation
# -----------------------------
def iterate_epoch(model, loader, criterion, optim=None, device="cpu", clip=1.0):
    is_train = optim is not None
    model.train(is_train)
    total_loss, total_tokens = 0.0, 0

    for x, y in tqdm(loader, disable=len(loader) < 10):
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)        # (B,T,V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if is_train:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()

        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, ppl

# -----------------------------
# 7) Generation (top-k/top-p)
# -----------------------------
def sample_next(probs: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """Nucleus/top-k sampling on a 1D probability tensor."""
    probs = probs.clone()
    # Top-k
    if top_k > 0:
        vals, idx = torch.topk(probs, k=min(top_k, probs.size(0)))
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[idx] = False
        probs[mask] = 0
        probs /= probs.sum()

    # Top-p (nucleus)
    if top_p > 0.0 and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum > top_p
        # Keep at least one token
        mask[0] = False
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum()
        # Map back
        probs = torch.zeros_like(probs)
        probs[sorted_idx] = sorted_probs

    dist = torch.distributions.Categorical(probs=probs)
    return dist.sample().item()

def generate_sentences(model, stoi, itos, prompt: str, device="cpu",
                       max_new_tokens=200, temperature=1.0, top_k=0, top_p=0.0,
                       num_sentences=1) -> str:
    model.eval()
    with torch.no_grad():
        tokens = tokenize(prompt, lower=True)
        bos_id = stoi[SPECIALS["BOS"]]
        eos_id = stoi[SPECIALS["EOS"]]
        unk_id = stoi[SPECIALS["UNK"]]
        x = torch.tensor([bos_id] + [stoi.get(t, unk_id) for t in tokens], dtype=torch.long, device=device).unsqueeze(0)

        hidden = None
        generated = tokens[:]  # words
        sentences_done = 0
        for _ in range(max_new_tokens):
            logits, hidden = model(x[:, -1:].contiguous(), hidden)  # feed one token at a time
            logits = logits[:, -1, :] / max(1e-6, temperature)
            probs = torch.softmax(logits.squeeze(0), dim=-1)
            nxt = sample_next(probs, top_k=top_k, top_p=top_p)
            word = itos[nxt]
            if word == SPECIALS["EOS"]:
                sentences_done += 1
                if sentences_done >= num_sentences:
                    break
                else:
                    generated.append(".")  # pretty print a dot between sentences
            elif word not in SPECIALS.values():
                generated.append(word)
            # Append next id to x
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)

        # Simple detokeniser: space before words, no space before certain punctuation
        out = []
        for i, w in enumerate(generated):
            if i > 0 and re.match(r"[^\w\s]", w):  # punctuation
                out[-1] = out[-1] + w
            else:
                out.append(w)
        return " ".join(out)

# -----------------------------
# 8) Corpus build
# -----------------------------
def load_corpus(books: List[str], data_dir="data", lower=True, min_freq=2) -> Tuple[List[int], List[int], dict, List[str]]:
    raw_dir = os.path.join(data_dir, "raw")
    txts = []
    for b in books:
        gid = GUTENBERG_IDS.get(b)
        if gid is None:
            print(f"[WARN] Unknown book key '{b}'. Skipping.")
            continue
        out_path = os.path.join(raw_dir, f"{b}.txt")
        ok = os.path.exists(out_path) or download_book(gid, out_path)
        if ok:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                txt = strip_gutenberg_boilerplate(f.read())
                txts.append(txt)
        else:
            print(f"[WARN] Failed to download book id={gid}.")
    if not txts:
        # Fallback tiny corpus so the pipeline always runs
        print("[INFO] Using tiny fallback corpus.")
        txts = ["""
            Once upon a time there was a small village by the sea. The wind was gentle.
            People watched the clouds and spoke softly. It was a quiet evening.
            The children played until the light faded, and lanterns glowed in the windows.
        """]

    # Combine, sentence-split, add EOS tokens between sentences
    all_sents = []
    for t in txts:
        all_sents.extend(sentence_split(t))
    # Tokenise + interleave EOS
    toks = []
    for s in all_sents:
        toks.extend(tokenize(s, lower=lower))
        toks.append(SPECIALS["EOS"])

    stoi, itos = build_vocab(toks, min_freq=min_freq)
    ids = numericalise([SPECIALS["BOS"]] + toks, stoi)

    # Train/val split (90/10) by contiguous stream
    n = len(ids)
    split = int(0.9 * n)
    train_ids = ids[:split]
    val_ids = ids[split:]
    return train_ids, val_ids, stoi, itos

# -----------------------------
# 9) Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Pure LSTM text completion (download + train + generate)")
    parser.add_argument("--books", type=str, default="alice,sherlock,pride",
                        help="Comma-separated keys from {alice,sherlock,pride,time_machine,dracula}")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--emb", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--lower", action="store_true", help="Lowercase the corpus")
    parser.add_argument("--no-lower", dest="lower", action="store_false")
    parser.set_defaults(lower=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save", type=str, default="lstm_lm.pt")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--clip", type=float, default=1.0)
    # Generation
    parser.add_argument("--generate", type=str, default="", help="Prompt to complete (skips training if provided with --resume)")
    parser.add_argument("--gen-sentences", type=int, default=1, help="How many sentences to generate after the prompt")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] Using device: {device}")

    # Load/build corpus
    books = [b.strip() for b in args.books.split(",") if b.strip()]
    train_ids, val_ids, stoi, itos = load_corpus(books, lower=args.lower, min_freq=args.min_freq)
    vocab_size = len(itos)
    print(f"[INFO] Vocab size: {vocab_size:,} | Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")

    # Datasets & loaders
    train_ds = LMStreamDataset(train_ids, args.seq_len)
    val_ds   = LMStreamDataset(val_ids, args.seq_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Model
    model = LSTMLanguageModel(vocab_size, emb=args.emb, hidden=args.hidden, layers=args.layers,
                              dropout=args.dropout, tie_weights=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Optionally resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        print(f"[INFO] Resumed from {args.resume} (epoch {ckpt.get('epoch','?')})")

    # If prompt given without training, we still need a trained model. Encourage resume.
    if args.generate and not args.resume:
        print("[WARN] --generate supplied but no --resume checkpoint given; "
              "you'll get untrained gibberish unless you train first.")

    # Train
    if args.epochs > 0:
        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            print(f"\n[Epoch {epoch}/{args.epochs}]")
            tr_loss, tr_ppl = iterate_epoch(model, train_dl, criterion, optim, device=device, clip=args.clip)
            va_loss, va_ppl = iterate_epoch(model, val_dl, criterion, None, device=device)

            print(f"  train loss: {tr_loss:.4f} | ppl: {tr_ppl:.2f}")
            print(f"  valid loss: {va_loss:.4f} | ppl: {va_ppl:.2f}")

            ckpt = {"model": model.state_dict(), "optim": optim.state_dict(),
                    "epoch": epoch, "vocab": {"stoi": stoi, "itos": itos}}
            torch.save(ckpt, args.save)

            if va_loss < best_val:
                best_val = va_loss
                torch.save(ckpt, os.path.splitext(args.save)[0] + ".best.pt")

    # Generate
    if args.generate:
        # If a checkpoint is provided (or we just trained), load latest weights for generation
        gen_ckpt_path = args.resume if args.resume else args.save
        if os.path.exists(gen_ckpt_path):
            ckpt = torch.load(gen_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            # prefer vocab from checkpoint if present
            if "vocab" in ckpt and "itos" in ckpt["vocab"] and "stoi" in ckpt["vocab"]:
                stoi = ckpt["vocab"]["stoi"]
                itos = ckpt["vocab"]["itos"]
                vocab_size = len(itos)
        else:
            print(f"[WARN] No checkpoint found at {gen_ckpt_path}; generating with current weights.")

        text = generate_sentences(
            model, stoi, itos, args.generate, device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_sentences=args.gen_sentences
        )
        print("\n=== Prompt ===")
        print(args.generate)
        print("\n=== Completion ===")
        print(text)
    else:
        print("\n[INFO] Training complete. Use --generate \"your prompt\" to produce next sentences.")

if __name__ == "__main__":
    main()
