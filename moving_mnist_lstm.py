# moving_mnist_lstm_all_in_one.py                                                     # Script filename for clarity and reproducibility

"""
All-in-one, lecture-ready demo showing LSTM sequence-to-sequence prediction.       # High-level description of the script
It supports two tasks:                                                              # Bullet list of supported tasks
1) MovingMNIST next-frame prediction (vision seq2seq).                              # Task 1 summary
2) Alphabet seq2seq with wrap-around (ABC→DEF, XYZ→YZA, A→B, E→F).                  # Task 2 summary
It also includes:                                                                   # Additional features
- Token-level cross-entropy for alphabet seq2seq (per-time-step supervision).       # Training objective for alphabet
- Tkinter GUI for interactive alphabet predictions (greedy by default).             # Interactive demo component
- Animated loss GIFs and a GT|Pred|Error triptych for qualitative evaluation.       # Visual outputs for teaching/inspection

Examples:                                                                            # How to run examples
  python moving_mnist_lstm_all_in_one.py --epochs 12 --down 32                      # Train MovingMNIST (default path)
  python moving_mnist_lstm_all_in_one.py --alphabet-mode --epochs 40                # Train alphabet seq2seq
  python moving_mnist_lstm_all_in_one.py --alphabet-gui                             # Launch interactive alphabet GUI
"""

from __future__ import annotations                                                  # Use postponed evaluation of annotations for cleaner type hints
import os, sys, time, json, argparse, subprocess, datetime as dt                    # Standard libraries for filesystem, CLI, timing and OS integration
from dataclasses import dataclass                                                   # Dataclass to store configuration cleanly
from typing import Optional, Tuple, List                                            # Type hints for readability and tooling
import numpy as np                                                                  # Numerical arrays and simple linear algebra
import torch, torch.nn as nn, torch.nn.functional as F                              # PyTorch: tensors, modules, and functional ops
from torch.utils.data import Dataset, DataLoader                                     # PyTorch dataset and data loader utilities

import matplotlib                                                                    # Matplotlib import to set backend before pyplot
matplotlib.use("Agg")                                                                # Use non-interactive backend (safe on servers/Windows)
import matplotlib.pyplot as plt                                                      # Pyplot for figures and animations
import imageio.v2 as imageio                                                         # ImageIO for writing GIFs

# ---------------------- Optional: torchvision for MovingMNIST -------------------
try:                                                                                 # Attempt to import torchvision dataset helpers
    from torchvision.datasets import MovingMNIST                                     # MovingMNIST dataset (downloadable)
    _HAS_TORCHVISION = True                                                          # Flag indicating availability
except Exception:                                                                     # If torchvision is not installed / available
    _HAS_TORCHVISION = False                                                         # Flag indicating unavailability

# ================================= Configuration ===============================

@dataclass                                                                            # Dataclass for structured configuration
class Cfg:
    root: str = "./data"                                                              # Download root for torchvision datasets
    cache_dir: str = "./data_cache"                                                   # Cache directory for processed NPZ files
    outdir: str = "./outputs"                                                         # Folder to store generated figures/GIFs
    weights_dir: str = "./weights"                                                    # Folder to store model checkpoints

    t_in: int = 10                                                                     # Number of observed frames (input length) for vision task
    t_out: int = 10                                                                    # Number of future frames to predict (output length) for vision task

    down: int = 32                                                                     # Spatial side length to which frames are downsampled (H=W)
    batch: int = 64                                                                    # Batch size used in training and validation
    hidden: int = 256                                                                  # Hidden size for the PixelLSTM model
    proj: int = 256                                                                    # Projection size from flattened pixels before LSTM
    epochs: int = 20                                                                   # Default number of epochs to train
    lr: float = 3e-3                                                                   # Learning rate for Adam optimiser

    max_train: Optional[int] = 2000                                                    # Cap on training sequences (None to use all)
    max_val: Optional[int] = 200                                                       # Cap on validation sequences (None to use all)
    num_workers: int = 2                                                               # DataLoader CPU workers (set 0 on Windows if issues)

    device: str = "auto"                                                               # Device choice: "auto" | "cuda" | "cpu" | "mps"
    seed: int = 123                                                                    # Random seed for reproducibility

    prepare_only: bool = False                                                         # If True, only prepare/cache dataset then exit
    resume_from: Optional[str] = None                                                  # Optional checkpoint path to resume from
    auto_open: bool = False                                                            # If True, attempt to open generated GIFs after training

    # Alphabet task configuration
    alphabet_mode: bool = False                                                        # If True, run alphabet training/eval instead of MovingMNIST
    alphabet_gui: bool = False                                                         # If True, launch the interactive alphabet GUI
    alphabet_seq_len: int = 6                                                          # Maximum training sequence length for alphabet data
    alphabet_embed_dim: int = 64                                                       # Embedding dimension for alphabet tokens
    alphabet_hidden_dim: int = 128                                                     # Hidden size of alphabet LSTM
    alphabet_layers: int = 2                                                           # Number of stacked LSTM layers for alphabet model

def parse_args() -> Cfg:                                                               # Parse command-line arguments and return a Cfg
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) # Create parser with defaults shown
    p.add_argument("--root", type=str, default=Cfg.root)                                # Dataset root flag
    p.add_argument("--cache-dir", type=str, default=Cfg.cache_dir)                      # Cache directory flag
    p.add_argument("--outdir", type=str, default=Cfg.outdir)                            # Outputs directory flag
    p.add_argument("--weights-dir", type=str, default=Cfg.weights_dir)                  # Weights directory flag
    p.add_argument("--t-in", type=int, default=Cfg.t_in, dest="t_in")                   # Observed steps flag
    p.add_argument("--t-out", type=int, default=Cfg.t_out, dest="t_out")                # Predicted steps flag
    p.add_argument("--down", type=int, default=Cfg.down)                                # Downsample size flag
    p.add_argument("--batch", type=int, default=Cfg.batch)                              # Batch size flag
    p.add_argument("--hidden", type=int, default=Cfg.hidden)                            # PixelLSTM hidden flag
    p.add_argument("--proj", type=int, default=Cfg.proj)                                # Pixel projection size flag
    p.add_argument("--epochs", type=int, default=Cfg.epochs)                            # Training epochs flag
    p.add_argument("--lr", type=float, default=Cfg.lr)                                  # Learning rate flag
    p.add_argument("--max-train", type=int, default=Cfg.max_train)                      # Max training sequences flag
    p.add_argument("--max-val", type=int, default=Cfg.max_val)                          # Max validation sequences flag
    p.add_argument("--num-workers", type=int, default=Cfg.num_workers)                  # DataLoader workers flag
    p.add_argument("--device", type=str, default=Cfg.device, choices=["auto","cuda","cpu","mps"])  # Device flag with choices
    p.add_argument("--seed", type=int, default=Cfg.seed)                                # Random seed flag
    p.add_argument("--prepare-only", action="store_true", dest="prepare_only")          # Flag to only prepare data
    p.add_argument("--resume-from", type=str, default=None, dest="resume_from")         # Checkpoint resume path flag
    p.add_argument("--auto-open", action="store_true", dest="auto_open")                # Auto-open outputs flag
    p.add_argument("--alphabet-mode", action="store_true", dest="alphabet_mode")        # Alphabet mode flag
    p.add_argument("--alphabet-gui", action="store_true", dest="alphabet_gui")          # Alphabet GUI flag
    p.add_argument("--alphabet-seq-len", type=int, default=Cfg.alphabet_seq_len, dest="alphabet_seq_len")     # Alphabet max seq length flag
    p.add_argument("--alphabet-embed-dim", type=int, default=Cfg.alphabet_embed_dim, dest="alphabet_embed_dim")# Alphabet embedding dim flag
    p.add_argument("--alphabet-hidden-dim", type=int, default=Cfg.alphabet_hidden_dim, dest="alphabet_hidden_dim")  # Alphabet hidden size flag
    p.add_argument("--alphabet-layers", type=int, default=Cfg.alphabet_layers, dest="alphabet_layers")         # Alphabet LSTM layers flag
    args = p.parse_args()                                                             # Parse provided CLI arguments
    return Cfg(**vars(args))                                                          # Convert argparse namespace to Cfg dataclass

# ================================== Utilities =================================

def seed_everything(seed: int) -> None:                                              # Set RNG seeds for reproducibility
    torch.manual_seed(seed)                                                          # Seed PyTorch RNG
    np.random.seed(seed)                                                             # Seed NumPy RNG

def pick_device(flag: str) -> torch.device:                                          # Select compute device based on flag/availability
    if flag == "cuda" and torch.cuda.is_available():                                 # If user requests CUDA and it's available
        return torch.device("cuda")                                                  # Return CUDA device
    if flag == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # If user requests MPS and it's available
        return torch.device("mps")                                                   # Return Apple MPS device
    if flag == "auto":                                                               # If auto selection requested
        if torch.cuda.is_available():                                                # Prefer CUDA if present
            return torch.device("cuda")                                              # Return CUDA
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): # Else prefer MPS if present
            return torch.device("mps")                                               # Return MPS
    return torch.device("cpu")                                                       # Default to CPU

def ensure_outdir(base: str) -> str:                                                 # Create a timestamped output subfolder and return its path
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")                                  # Format current timestamp
    od = os.path.join(base, ts)                                                       # Join base path with timestamp
    os.makedirs(od, exist_ok=True)                                                    # Create directory if missing
    return od                                                                         # Return created path

def maybe_open(path: str) -> None:                                                    # Try to open a file with the OS default app
    try:                                                                              # Wrap in try/except to be safe cross-platform
        if os.name == "nt": os.startfile(path)                                        # Windows open
        elif sys.platform == "darwin": subprocess.Popen(["open", path])               # macOS open
        else: subprocess.Popen(["xdg-open", path])                                     # Linux open
    except Exception:                                                                  # If anything fails
        pass                                                                           # Silently ignore (non-fatal)

# =========================== MovingMNIST data pipeline =========================

def cache_key(down: int) -> str:                                                      # Generate a cache filename based on downsample size
    return f"moving_mnist_all_down{down}.npz"                                         # Stable naming for reuse

def prepare_and_cache(cfg: Cfg) -> str:                                               # Prepare MovingMNIST and cache as NPZ
    if not _HAS_TORCHVISION:                                                          # If torchvision is unavailable
        raise RuntimeError("torchvision not available; use --alphabet-mode")          # Inform user to use alphabet mode
    os.makedirs(cfg.cache_dir, exist_ok=True)                                         # Create cache directory if needed
    cache_path = os.path.join(cfg.cache_dir, cache_key(cfg.down))                     # Build full cache path
    if os.path.isfile(cache_path):                                                    # If cache exists
        print(f"[Data] Using cached NPZ at {cache_path}")                             # Inform reuse
        return cache_path                                                             # Return cached path
    print("[Data] Downloading/processing MovingMNIST ...")                            # Log dataset acquisition
    try:                                                                              # Try modern API signature
        all_raw = MovingMNIST(root=cfg.root, split=None, download=True)               # Download dataset
    except TypeError:                                                                  # Fallback for older torchvision
        all_raw = MovingMNIST(root=cfg.root, download=True)                           # Alternative constructor
    data = getattr(all_raw, "data")                                                   # Access raw array/tensor
    tensor = torch.from_numpy(data) if isinstance(data, np.ndarray) else data         # Convert to tensor if needed
    if tensor.dim() == 5: tensor = tensor.squeeze(2)                                  # If (N,T,1,H,W) → (N,T,H,W)
    elif tensor.dim() == 4 and tensor.shape[0] <= 30 and tensor.shape[1] > 30:        # If stored as (T,N,H,W)
        tensor = tensor.permute(1,0,2,3)                                              # Transpose to (N,T,H,W)
    N,T,H,W = tensor.shape                                                            # Unpack shapes
    NT = N*T                                                                          # Total frames
    x = tensor.reshape(NT,1,H,W).float()                                              # Flatten sequence dimension and add channel
    x = F.interpolate(x, size=(cfg.down,cfg.down), mode="bilinear", align_corners=False)  # Downsample with bilinear
    x = x.reshape(N,T,1,cfg.down,cfg.down).clamp(0,255).round().to(torch.uint8)       # Reassemble and cast to uint8
    arr = x.squeeze(2).cpu().numpy()                                                  # Remove channel and convert to numpy (N,T,H,W)
    print(f"[Data] Saving NPZ -> {cache_path}")                                       # Log caching
    np.savez_compressed(cache_path, all=arr)                                          # Save compressed NPZ
    return cache_path                                                                  # Return path to NPZ

def load_processed_npz(cache_path: str) -> np.ndarray:                                # Load cached MovingMNIST NPZ
    d = np.load(cache_path)                                                           # Open NPZ file
    return d["all"]                                                                   # Return array under key "all"

class SeqPredictDataset(Dataset):                                                     # Dataset: returns (x,y) where y is future frames
    def __init__(self, arr_uint8: np.ndarray, t_in: int, t_out: int):                 # Store data and lengths
        self.arr, self.t_in, self.t_out = arr_uint8, t_in, t_out                      # Assign members
        assert self.arr.shape[1] >= (t_in + t_out), "Need T >= t_in + t_out"          # Sanity check on sequence length
    def __len__(self) -> int: return self.arr.shape[0]                                # Number of sequences available
    def __getitem__(self, idx: int):                                                  # Fetch a sample by index
        seq = self.arr[idx]                                                           # Retrieve sequence array (T,H,W)
        x = torch.from_numpy(seq[:self.t_in]).float().div_(255.0).unsqueeze(1)        # Normalise input frames and add channel dim → (T_in,1,H,W)
        y = torch.from_numpy(seq[self.t_in:self.t_in+self.t_out]).float().div_(255.0).unsqueeze(1)  # Same for targets → (T_out,1,H,W)
        return x, y                                                                    # Return tensors

# ================================ Alphabet data ================================

ALPHABET = [chr(ord('A') + i) for i in range(26)]                                     # List of uppercase letters A–Z

def _clean_letters(s: str) -> str:                                                    # Remove non-letters and uppercase the input
    return "".join(c for c in s.upper() if 'A' <= c <= 'Z')                           # Keep A–Z only

def letters_to_idx(seq: str) -> List[int]:                                            # Convert string of letters to list of indices 0..25
    seq = _clean_letters(seq)                                                         # Clean input
    return [ord(c) - 65 for c in seq]                                                 # Convert each char to index

def idx_to_letters(idxs: List[int]) -> str:                                           # Convert list of indices back to a string
    return "".join(ALPHABET[i % 26] for i in idxs)                                    # Wrap modulo-26 just in case

def generate_alphabet_data_seq2seq(max_seq_len: int = 6, num_samples: int = 20000, min_seq_len: int = 2):  # Generate synthetic seq2seq pairs
    rng = np.random.default_rng(12345)                                                # Deterministic RNG for reproducibility
    data: List[Tuple[List[int], List[int]]] = []                                      # Storage for (X,Y) integer pairs
    for _ in range(num_samples):                                                      # Loop to create samples
        L = int(rng.integers(min_seq_len, max_seq_len + 1))                           # Sample a sequence length uniformly
        start = int(rng.integers(0, 26))                                              # Sample a starting letter index
        x = [(start + t) % 26 for t in range(L)]                                      # Build consecutive sequence (wrap-around)
        y = [(xi + 1) % 26 for xi in x]                                               # Targets are +1 shifted (wrap-around)
        data.append((x, y))                                                           # Append the pair
    return data                                                                       # Return all pairs

class AlphabetSeq2SeqDataset(Dataset):                                                # Dataset that returns variable-length integer sequences
    def __init__(self, pairs: List[Tuple[List[int], List[int]]]):                     # Constructor stores (X,Y) pairs
        self.pairs = pairs                                                            # Save pairs
    def __len__(self) -> int: return len(self.pairs)                                  # Return dataset size
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:               # Return tensors for one sample
        x, y = self.pairs[i]                                                          # Get (X,Y)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)   # Convert to int64 tensors

def pad_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):                        # Collate function to pad variable-length sequences
    xs, ys = zip(*batch)                                                              # Separate X and Y lists
    maxL = max(x.size(0) for x in xs)                                                 # Find maximum sequence length in the batch
    X = torch.zeros(len(xs), maxL, dtype=torch.long)                                  # Allocate padded X with zeros ('A' as pad index)
    Y = torch.zeros(len(xs), maxL, dtype=torch.long)                                  # Allocate padded Y similarly
    M = torch.zeros(len(xs), maxL, dtype=torch.bool)                                  # Boolean mask: True where tokens are real
    for i, (x, y) in enumerate(zip(xs, ys)):                                          # Iterate over samples
        L = x.size(0)                                                                 # Length of this sample
        X[i, :L] = x                                                                  # Copy X into padded tensor
        Y[i, :L] = y                                                                  # Copy Y similarly
        M[i, :L] = True                                                               # Mark valid positions in mask
    return X, Y, M                                                                     # Return padded batch and mask

# =================================== Models ===================================

class PixelLSTM(nn.Module):                                                           # Vision seq2seq model: frame sequence → frame sequence
    def __init__(self, h: int, w: int, proj: int, hidden: int):                       # Constructor with sizes
        super().__init__()                                                            # Initialise base nn.Module
        self.h, self.w = h, w                                                         # Store spatial dimensions
        self.npix = h * w                                                             # Number of pixels per frame
        self.proj = nn.Linear(self.npix, proj)                                        # Linear projection from flattened frame to latent
        self.lstm = nn.LSTM(proj, hidden, num_layers=1, batch_first=True)             # Single-layer LSTM over time
        self.head = nn.Linear(hidden, self.npix)                                      # Map hidden state back to pixel space
    def forward(self, x: torch.Tensor) -> torch.Tensor:                               # Forward pass for training/inference
        B,T,C,H,W = x.shape                                                           # Unpack shape
        z = self.proj(x.view(B, T, -1))                                               # Flatten frames and project → (B,T,proj)
        out, _ = self.lstm(z)                                                         # Pass through LSTM across time
        y = self.head(out).view(B, T, 1, self.h, self.w)                               # Project back and reshape to images
        return torch.sigmoid(y)                                                       # Sigmoid to keep pixels in [0,1]
    @torch.no_grad()                                                                  # Disable gradients for generation
    def predict_future(self, x_obs: torch.Tensor, steps: int) -> torch.Tensor:        # Autoregressive prediction of future frames
        self.eval()                                                                    # Use eval mode for deterministic behaviour
        B,T,C,H,W = x_obs.shape                                                        # Unpack shape
        z = self.proj(x_obs.view(B, T, -1))                                           # Project observed frames
        _, (h, c) = self.lstm(z)                                                      # Encode history to hidden states
        last = x_obs[:, -1]                                                           # Start from the last observed frame
        preds: List[torch.Tensor] = []                                                # Collect predicted frames
        for _ in range(steps):                                                        # Loop for each future step
            zt = self.proj(last.view(B, -1)).unsqueeze(1)                             # Project last frame and add time dim
            out, (h, c) = self.lstm(zt, (h, c))                                       # One-step LSTM update
            y = torch.sigmoid(self.head(out[:, -1]).view(B, 1, H, W))                 # Map to pixel grid with sigmoid
            preds.append(y.unsqueeze(1))                                              # Append with time dim
            last = y                                                                   # Feed prediction as next input
        return torch.cat(preds, dim=1)                                                # Concatenate predictions over time

class AlphabetLSTM(nn.Module):                                                        # Alphabet seq2seq model: token sequence → next-token sequence
    def __init__(self, vocab_size: int = 26, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2):  # Constructor
        super().__init__()                                                            # Call base initialiser
        self.embed = nn.Embedding(vocab_size, embed_dim)                              # Embedding layer for discrete tokens
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)  # Stacked LSTM
        self.head  = nn.Linear(hidden_dim, vocab_size)                                # Linear head to vocab logits
    def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # Forward pass
        z = self.embed(x)                                                             # Look up embeddings for each token
        h, hidden = self.lstm(z, hidden)                                              # Run LSTM over time
        logits = self.head(h)                                                         # Project hidden states to vocabulary logits
        return logits, hidden                                                         # Return logits and (h,c)
    @torch.no_grad()                                                                  # No grad for generation
    def generate(self, seed: str, steps: int, device=None, temperature: float = 0.0) -> str:  # Autoregressive generation from seed
        idx = letters_to_idx(seed) or [0]                                             # Convert seed to indices (default to 'A' if empty)
        x = torch.tensor([idx], dtype=torch.long, device=device)                      # Create batch-1 tensor
        _, hidden = self.forward(x)                                                   # Warm-up LSTM state with seed
        last = x[:, -1:]                                                              # Start generation from last token
        out_ids: List[int] = []                                                       # List to collect generated ids
        for _ in range(max(1, steps)):                                                # Generate required number of steps
            logits, hidden = self.forward(last, hidden)                               # One-step forward pass
            logits = logits[:, -1, :]                                                 # Get logits for the generated step
            if temperature and temperature > 0:                                       # If sampling with temperature
                probs = torch.softmax(logits / temperature, dim=-1)                   # Convert to probabilities
                nxt = torch.multinomial(probs, 1)                                     # Sample next token index
            else:                                                                      # Otherwise greedy (deterministic)
                nxt = torch.argmax(logits, dim=-1, keepdim=True)                      # Argmax selection
            out_ids.append(int(nxt.item()))                                           # Append selected id
            last = nxt                                                                 # Feed as next input
        return idx_to_letters(out_ids)                                                # Convert generated ids back to letters
    @torch.no_grad()                                                                  # Helper with same-horizon as input length
    def predict_same_length(self, seed: str, device=None, temperature: float = 0.0) -> str:  # Same-length generation
        return self.generate(seed, steps=len(_clean_letters(seed)), device=device, temperature=temperature)  # Delegate

# ============================= Alphabet losses/metrics =========================

def seq2seq_token_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # Compute masked per-token CE
    V = logits.size(-1)                                                              # Vocabulary size
    ce = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='none')     # Token CE without reduction
    ce = ce.view_as(mask)                                                            # Reshape to (B,T)
    ce = ce[mask].mean()                                                             # Average over valid positions
    return ce                                                                        # Return scalar loss

@torch.no_grad()                                                                     # No gradients for metric computation
def seq2seq_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[float,float]:  # Compute token/seqlen accuracies
    pred = logits.argmax(dim=-1)                                                     # Predicted token indices
    correct_tokens = ((pred == targets) & mask).sum().item()                          # Count correct where mask is True
    total_tokens   = mask.sum().item()                                               # Count total valid tokens
    token_acc = correct_tokens / max(1, total_tokens)                                # Token-level accuracy
    exact_flags: List[bool] = []                                                     # Container for exact-sequence matches
    for i in range(mask.size(0)):                                                    # Iterate over batch samples
        L = int(mask[i].sum().item())                                                # Valid length for this sample
        exact_flags.append(bool((pred[i, :L] == targets[i, :L]).all().item()))        # Exact match flag
    seq_acc = sum(exact_flags) / max(1, len(exact_flags))                            # Exact-sequence accuracy
    return token_acc, seq_acc                                                        # Return both metrics

# ============================== Training functions =============================

def train_one_epoch(model: nn.Module, opt: torch.optim.Optimizer, loader: DataLoader, device: torch.device) -> Tuple[float,float]:  # Vision training epoch
    model.train()                                                                    # Enable training mode
    mse = nn.MSELoss()                                                               # Use MSE for pixel regression
    loss_sum, grad_sum, steps = 0.0, 0.0, 0                                          # Initialise accumulators
    for x, y in loader:                                                              # Iterate over mini-batches
        x, y = x.to(device), y.to(device)                                            # Move tensors to device
        opt.zero_grad(set_to_none=True)                                              # Reset gradients
        y_hat = model(x)                                                             # Forward pass
        T = min(y_hat.shape[1], y.shape[1])                                          # Align sequence lengths
        loss = mse(y_hat[:, :T], y[:, :T])                                           # Compute MSE loss over aligned steps
        loss.backward()                                                              # Backpropagate
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)                  # Clip gradients to avoid explosion
        opt.step()                                                                   # Update parameters
        loss_sum += float(loss.item())                                               # Accumulate loss
        grad_sum += float(g)                                                         # Accumulate grad norm
        steps += 1                                                                   # Increment step count
    return loss_sum/max(1, steps), grad_sum/max(1, steps)                            # Return average loss and grad norm

@torch.no_grad()                                                                     # No gradients for evaluation
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:   # Vision validation pass
    model.eval()                                                                     # Switch to eval mode
    mse = nn.MSELoss()                                                               # Same loss as training
    run = 0.0                                                                        # Accumulator for loss
    for x, y in loader:                                                              # Iterate batches
        x, y = x.to(device), y.to(device)                                            # Move to device
        y_hat = model(x)                                                             # Forward
        T = min(y_hat.shape[1], y.shape[1])                                          # Align timesteps
        run += float(mse(y_hat[:, :T], y[:, :T]).item())                             # Add batch loss
    return run/max(1, len(loader))                                                   # Return mean loss

def train_alphabet_epoch_seq2seq(model: nn.Module, opt: torch.optim.Optimizer, loader: DataLoader, device: torch.device) -> Tuple[float,float,float]:  # Alphabet training epoch
    model.train()                                                                    # Train mode
    loss_sum = 0.0                                                                   # Loss accumulator
    token_acc_sum = 0.0                                                              # Token-accuracy accumulator
    seq_acc_sum = 0.0                                                                # Sequence-accuracy accumulator
    steps = 0                                                                        # Step counter
    for X, Y, M in loader:                                                           # Batches of (inputs, targets, mask)
        X, Y, M = X.to(device), Y.to(device), M.to(device)                           # Move to device
        opt.zero_grad(set_to_none=True)                                              # Zero grads
        logits, _ = model(X)                                                         # Predict logits at each timestep
        loss = seq2seq_token_loss(logits, Y, M)                                      # Compute masked token-level CE
        loss.backward()                                                              # Backpropagate
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)                      # Clip gradients
        opt.step()                                                                   # Optimiser update
        tok_acc, seq_acc = seq2seq_accuracy(logits.detach(), Y, M)                   # Compute accuracies
        loss_sum += float(loss.item())                                               # Accumulate loss
        token_acc_sum += tok_acc                                                     # Accumulate token acc
        seq_acc_sum += seq_acc                                                       # Accumulate sequence acc
        steps += 1                                                                   # Increment steps
    return (loss_sum/steps, token_acc_sum/steps, seq_acc_sum/steps)                  # Return averaged metrics

@torch.no_grad()                                                                     # Evaluation does not require grads
def evaluate_alphabet_seq2seq(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float,float,float]:  # Alphabet validation epoch
    model.eval()                                                                     # Eval mode
    loss_sum = 0.0                                                                   # Loss accumulator
    token_acc_sum = 0.0                                                              # Token accuracy accumulator
    seq_acc_sum = 0.0                                                                # Sequence accuracy accumulator
    steps = 0                                                                        # Step counter
    for X, Y, M in loader:                                                           # Iterate over val batches
        X, Y, M = X.to(device), Y.to(device), M.to(device)                           # Move to device
        logits, _ = model(X)                                                         # Forward
        loss = seq2seq_token_loss(logits, Y, M)                                      # Loss
        tok_acc, seq_acc = seq2seq_accuracy(logits, Y, M)                            # Accuracies
        loss_sum += float(loss.item())                                               # Accumulate loss
        token_acc_sum += tok_acc                                                     # Accumulate token acc
        seq_acc_sum += seq_acc                                                       # Accumulate seq acc
        steps += 1                                                                    # Increment step count
    return (loss_sum/steps, token_acc_sum/steps, seq_acc_sum/steps)                  # Return averaged metrics

# ================================= Visualisation ===============================

def _ema(arr: np.ndarray, alpha: float=0.7) -> np.ndarray:                           # Exponential moving average smoother
    if len(arr) == 0: return arr                                                     # Edge case: empty input
    out = np.zeros_like(arr, dtype=float)                                            # Allocate output array
    out[0] = arr[0]                                                                  # Seed first value
    for i in range(1, len(arr)):                                                     # Iterate remaining indices
        out[i] = alpha*out[i-1] + (1-alpha)*arr[i]                                   # EMA recurrence
    return out                                                                        # Return smoothed series

def animate_losses(train_hist: List[float], val_hist: List[float], out_path: str, smooth_alpha: float=0.75) -> None:  # Animate training/val loss
    t = np.asarray(train_hist, dtype=float)                                          # Convert train history to array
    v = np.asarray(val_hist, dtype=float)                                            # Convert val history to array
    ts = _ema(t, smooth_alpha)                                                       # Smooth train series
    vs = _ema(v, smooth_alpha)                                                       # Smooth val series
    frames: List[np.ndarray] = []                                                    # Container for rendered frames
    fig, ax = plt.subplots(figsize=(6.6, 4.2))                                       # Create figure/axes
    for i in range(1, len(ts)+1):                                                    # Progressive animation frames
        ax.cla()                                                                      # Clear axes
        ax.set_title("Training & Validation Loss (EMA)")                             # Title
        ax.set_xlabel("Epoch")                                                       # X label
        ax.set_ylabel("Loss")                                                        # Y label
        ax.plot(ts[:i], label="train (smoothed)")                                    # Plot partial train curve
        ax.plot(vs[:i], label="val (smoothed)")                                      # Plot partial val curve
        ax.scatter(np.arange(len(t)), t, s=10, alpha=0.2, label="train raw")         # Scatter raw train points
        ax.scatter(np.arange(len(v)), v, s=10, alpha=0.2, label="val raw")           # Scatter raw val points
        ax.legend(loc="best")                                                        # Legend placement
        fig.canvas.draw()                                                            # Render to buffer
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())          # Copy RGBA buffer
    plt.close(fig)                                                                    # Close figure to free memory
    imageio.mimsave(out_path, frames, duration=0.08)                                 # Save animation as GIF

def prediction_triptych_gif(gt_seq: np.ndarray, pr_seq: np.ndarray, out_path: str) -> None:  # Create GT|Pred|Error triptych GIF
    assert gt_seq.shape == pr_seq.shape                                              # Ensure same shapes
    T,H,W = gt_seq.shape                                                             # Unpack dims
    gt = (np.clip(gt_seq,0,1)*255).astype(np.uint8)                                  # Scale GT to [0,255] uint8
    pr = (np.clip(pr_seq,0,1)*255).astype(np.uint8)                                  # Scale Pred to [0,255] uint8
    err = np.abs(gt.astype(np.int16)-pr.astype(np.int16)).astype(np.uint8)           # Absolute error map uint8
    frames: List[np.ndarray] = []                                                    # Frame buffer
    for t in range(T):                                                               # For each timestep
        fig, axes = plt.subplots(1,3,figsize=(9.2,3.2))                              # Three-panel figure
        for ax in axes: ax.axis('off')                                               # Hide axes for a clean look
        mse_t = float((((gt[t].astype(np.float32)/255.0)-(pr[t].astype(np.float32)/255.0))**2).mean())  # Compute per-frame MSE
        axes[0].imshow(gt[t], cmap='gray', vmin=0, vmax=255); axes[0].set_title('Ground Truth')          # Show GT
        axes[1].imshow(pr[t], cmap='gray', vmin=0, vmax=255); axes[1].set_title('Prediction')            # Show Pred
        im = axes[2].imshow(err[t], cmap='inferno', vmin=0.0, vmax=float(err.max())); axes[2].set_title(f'|Error|  MSE={mse_t:.4f}')  # Show Error
        cax = fig.add_axes([0.92,0.2,0.015,0.6]); plt.colorbar(im, cax=cax)          # Add colour bar for error magnitude
        fig.tight_layout(rect=[0,0,0.90,1])                                          # Adjust layout to fit colour bar
        fig.canvas.draw()                                                            # Render frame
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())          # Append numpy copy
        plt.close(fig)                                                               # Close to free memory
    imageio.mimsave(out_path, frames, duration=0.10)                                 # Save as GIF

# ================================ Checkpointing ================================

def save_checkpoint(model: nn.Module, outdir: str, train_hist: List[float], val_hist: List[float]) -> str:  # Save weights and history
    ckpt = os.path.join(outdir, "model.pt")                                          # Compose checkpoint path
    torch.save({"state_dict": model.state_dict(), "train_hist": train_hist, "val_hist": val_hist}, ckpt)  # Save model and histories
    with open(os.path.join(outdir, "history.json"), "w") as f:                       # Open JSON file for loss curves
        json.dump({"train": train_hist, "val": val_hist}, f)                         # Dump histories to JSON
    return ckpt                                                                       # Return checkpoint path

def load_checkpoint(model: nn.Module, path: str) -> Tuple[List[float], List[float]]: # Load checkpoint and return histories
    obj = torch.load(path, map_location="cpu")                                       # Load on CPU to be safe
    model.load_state_dict(obj["state_dict"])                                         # Restore weights into model
    return obj.get("train_hist",[]), obj.get("val_hist",[])                          # Return stored histories (or empty lists)

# ---------------------- Robust loader for legacy alphabet weights --------------
def load_alphabet_weights_with_remap(model: nn.Module, path: str, device: torch.device):  # Helper to load old/new key names safely
    """
    Load an alphabet model checkpoint while handling:
    - Checkpoints that wrap weights in {'state_dict': ...}
    - Legacy key names ('embedding.' → 'embed.', 'fc.' → 'head.')
    - Newer PyTorch safe loading (weights_only=True) when available
    """
    try:                                                                              # Try safer loading (if torch supports it)
        sd = torch.load(path, map_location=device, weights_only=True)                # PyTorch >= 2.5
    except TypeError:                                                                 # Older torch versions
        sd = torch.load(path, map_location=device)                                   # Fall back to classic load

    if isinstance(sd, dict) and 'state_dict' in sd:                                  # If a full checkpoint dict
        sd = sd['state_dict']                                                        # Extract the state dict

    remapped = {}                                                                     # Dictionary for remapped keys
    for k, v in sd.items():                                                           # Iterate over all keys
        nk = k.replace('embedding.', 'embed.').replace('fc.', 'head.')               # Map legacy names to current module names
        remapped[nk] = v                                                             # Store in new dict

    missing, unexpected = model.load_state_dict(remapped, strict=False)              # Load non-strict so tiny diffs won't crash
    if missing:    print("[load] missing keys:", missing)                            # Print any missing keys (informational)
    if unexpected: print("[load] unexpected keys:", unexpected)                      # Print any unexpected keys (informational)

# =================================== GUI ======================================

def run_alphabet_gui(model: AlphabetLSTM, device: torch.device) -> None:             # Launch interactive alphabet predictor
    try:                                                                              # Try to import Tkinter
        import tkinter as tk                                                          # GUI toolkit
        from tkinter import ttk, messagebox                                           # Widgets and dialogs
        model.eval()                                                                  # Evaluation mode for stable predictions
        root = tk.Tk()                                                                # Create main window
        root.title("Alphabet Predictor (Seq2Seq LSTM)")                               # Set window title
        root.geometry("560x340")                                                      # Set window size
        frm = ttk.Frame(root, padding=12)                                             # Create frame container
        frm.pack(fill="both", expand=True)                                            # Pack it to fill window
        ttk.Label(frm, text="Seed sequence (letters only):").grid(row=0, column=0, sticky="w")  # Label for input
        seed_var = tk.StringVar(value="ABC")                                          # Default seed value
        seed_entry = ttk.Entry(frm, textvariable=seed_var, width=40)                  # Text entry widget
        seed_entry.grid(row=0, column=1, columnspan=3, sticky="we", padx=(6,0))       # Place entry in grid
        ttk.Label(frm, text="Steps:").grid(row=1, column=0, sticky="w", pady=(8,0))   # Label for steps
        steps_var = tk.IntVar(value=3)                                                # Default steps (3 → ABC→DEF)
        steps_spin = ttk.Spinbox(frm, from_=1, to=26, textvariable=steps_var, width=6) # Spinbox for steps
        steps_spin.grid(row=1, column=1, sticky="w", padx=(6,12), pady=(8,0))         # Place spinbox
        match_len_var = tk.BooleanVar(value=True)                                     # Boolean to match input length
        ttk.Checkbutton(frm, text="Match input length", variable=match_len_var).grid(row=1, column=2, sticky="w", pady=(8,0))  # Checkbox
        ttk.Label(frm, text="Temperature:").grid(row=2, column=0, sticky="w", pady=(8,0))  # Label for temperature
        temp_var = tk.DoubleVar(value=0.0)                                            # Default 0.0 → greedy deterministic
        ttk.Scale(frm, from_=0.0, to=1.5, orient="horizontal", variable=temp_var).grid(row=2, column=1, columnspan=2, sticky="we", padx=(6,0), pady=(8,0))  # Slider
        ttk.Separator(frm).grid(row=3, column=0, columnspan=4, sticky="we", pady=10)  # Visual separator
        out_box = tk.Text(frm, height=10, width=64)                                   # Text box for outputs
        out_box.grid(row=4, column=0, columnspan=4, sticky="nsew")                    # Place output box
        frm.rowconfigure(4, weight=1); frm.columnconfigure(3, weight=1)               # Make output box expandable

        def predict(evt=None):                                                        # Handler for prediction button/Enter key
            seed = _clean_letters(seed_var.get())                                     # Clean input
            if not seed:                                                              # Validate non-empty
                return messagebox.showwarning("Input required", "Please enter letters.")  # Warn user
            steps = len(seed) if match_len_var.get() else max(1, int(steps_var.get())) # Decide number of steps
            pred = model.generate(seed, steps=steps, device=device, temperature=float(temp_var.get()))  # Generate prediction
            out_box.insert("end", f"{seed} → {pred}\n"); out_box.see("end")           # Append to output box

        ttk.Button(frm, text="Predict", command=predict).grid(row=5, column=0, columnspan=4, pady=(8,0))  # Predict button
        seed_entry.bind("<Return>", predict)                                           # Bind Enter key to predict
        root.mainloop()                                                                # Start GUI loop

    except Exception:                                                                  # If Tkinter not available (e.g., headless)
        print("\nAlphabet Seq2Seq (console). Type 'quit' to exit.\n")                 # Console fallback header
        model.eval()                                                                   # Eval mode
        while True:                                                                    # REPL loop
            s = input("Seed: ").strip()                                               # Read input
            if s.lower() in ("q","quit","exit"): break                                 # Exit on command
            seed = _clean_letters(s)                                                   # Clean input to A–Z
            if not seed:                                                               # Validate
                print("Please enter A–Z letters only.\n"); continue                    # Prompt again
            pred = model.predict_same_length(seed, device=device, temperature=0.0)     # Predict same length greedily
            print(f"  {seed} → {pred}\n")                                              # Show result

# =============================== Alphabet runner ===============================

def run_alphabet_training(cfg: Cfg, device: torch.device) -> None:                   # Train/evaluate alphabet seq2seq end-to-end
    print("🔤 Training Alphabet Seq2Seq (wrap-around, token-level CE)")              # Informative header
    train_pairs = generate_alphabet_data_seq2seq(max_seq_len=cfg.alphabet_seq_len, num_samples=16000, min_seq_len=2)  # Build training pairs
    val_pairs   = generate_alphabet_data_seq2seq(max_seq_len=cfg.alphabet_seq_len, num_samples=2000,  min_seq_len=2)  # Build validation pairs
    train_loader = DataLoader(AlphabetSeq2SeqDataset(train_pairs), batch_size=cfg.batch, shuffle=True,  num_workers=cfg.num_workers, collate_fn=pad_batch)  # Train loader
    val_loader   = DataLoader(AlphabetSeq2SeqDataset(val_pairs),   batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, collate_fn=pad_batch)  # Val loader
    model = AlphabetLSTM(vocab_size=26, embed_dim=cfg.alphabet_embed_dim, hidden_dim=cfg.alphabet_hidden_dim, num_layers=cfg.alphabet_layers).to(device)    # Create model
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)                             # Adam optimiser
    outdir = ensure_outdir(cfg.outdir)                                                # Create timestamped output directory
    weights_dir = os.path.join(cfg.weights_dir, os.path.basename(outdir))             # Mirror weights folder
    os.makedirs(weights_dir, exist_ok=True)                                           # Ensure weights dir exists
    train_loss_hist: List[float] = []                                                 # History list for train loss
    val_loss_hist:   List[float] = []                                                 # History list for val loss
    bl_loss, bl_tok, bl_seq = evaluate_alphabet_seq2seq(model, val_loader, device)    # Baseline (untrained) metrics
    train_loss_hist.append(bl_loss); val_loss_hist.append(bl_loss)                    # Seed histories with baseline
    print(f"Baseline | val CE {bl_loss:.4f} | token acc {bl_tok:.3f} | seq acc {bl_seq:.3f}")  # Print baseline
    for epoch in range(1, cfg.epochs + 1):                                            # Training loop
        t0 = time.time()                                                              # Start timer
        tr_loss, tr_tok, tr_seq = train_alphabet_epoch_seq2seq(model, opt, train_loader, device)  # Train for one epoch
        va_loss, va_tok, va_seq = evaluate_alphabet_seq2seq(model, val_loader, device)            # Validate epoch
        train_loss_hist.append(tr_loss); val_loss_hist.append(va_loss)                # Record losses
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train CE {tr_loss:.4f} tok {tr_tok:.3f} seq {tr_seq:.3f} | val CE {va_loss:.4f} tok {va_tok:.3f} seq {va_seq:.3f} | {time.time()-t0:.1f}s")  # Log progress
    ckpt_path = save_checkpoint(model, weights_dir, train_loss_hist, val_loss_hist)   # Save checkpoint + histories
    print("\n🧪 Sanity predictions (greedy):")                                        # Header for quick tests
    for s in ["A", "E", "Z", "AB", "LMN", "XYZ", "ABC"]:                              # Example seeds
        print(f"  {s:>3} → {model.predict_same_length(s, device=device, temperature=0.0)}")  # Show deterministic predictions
    loss_gif = os.path.join(outdir, "alphabet_loss.gif")                              # Path for loss GIF
    animate_losses(train_loss_hist, val_loss_hist, loss_gif)                          # Create loss animation
    print(f"\nOutdir:     {outdir}\nWeights:    {weights_dir}\nCheckpoint: {ckpt_path}\nLoss GIF:   {loss_gif}")  # Summarise outputs
    if cfg.auto_open: maybe_open(loss_gif)                                            # Optionally open loss GIF

# ==================================== Main ====================================

def main() -> None:                                                                  # Entry point
    cfg = parse_args()                                                               # Parse CLI to config
    seed_everything(cfg.seed)                                                        # Set RNG seeds
    device = pick_device(cfg.device)                                                 # Choose compute device
    print(f"Using device: {device}")                                                 # Log device

    if cfg.alphabet_gui:                                                             # If GUI requested
        model = AlphabetLSTM(vocab_size=26, embed_dim=cfg.alphabet_embed_dim, hidden_dim=cfg.alphabet_hidden_dim, num_layers=cfg.alphabet_layers).to(device)  # Build model
        quick_path = os.path.join(cfg.weights_dir, "alphabet_model.pth")             # Path for quick GUI weights
        if os.path.exists(quick_path):                                               # If a saved model exists
            print(f"Loading pre-trained alphabet model from {quick_path}")           # Inform user
            load_alphabet_weights_with_remap(model, quick_path, device)              # Load safely with legacy-name remap
        else:                                                                         # Otherwise do a quick warm-up train
            print("No pre-trained model found. Running a short warm-up training...") # Inform user
            pairs = generate_alphabet_data_seq2seq(max_seq_len=cfg.alphabet_seq_len, num_samples=4000, min_seq_len=2)  # Small dataset
            loader = DataLoader(AlphabetSeq2SeqDataset(pairs), batch_size=cfg.batch, shuffle=True, collate_fn=pad_batch)  # Loader
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)                    # Optimiser
            for e in range(15):                                                      # Train for a few epochs
                tr, tok, seq = train_alphabet_epoch_seq2seq(model, opt, loader, device)  # Train epoch
                if e % 5 == 0: print(f"  warm-up epoch {e:02d} | CE {tr:.4f} | tok {tok:.3f} | seq {seq:.3f}")  # Progress
            os.makedirs(cfg.weights_dir, exist_ok=True)                               # Ensure directory
            torch.save(model.state_dict(), quick_path)                                # Save quick model (raw state_dict)
            print(f"Saved warm-up model to {quick_path}")                             # Log path
        run_alphabet_gui(model, device)                                              # Launch GUI
        return                                                                        # Exit after GUI

    if cfg.alphabet_mode:                                                            # If alphabet training requested
        run_alphabet_training(cfg, device)                                           # Run alphabet training pipeline
        return                                                                        # Exit when done

    if not _HAS_TORCHVISION:                                                         # If default path but no torchvision
        raise RuntimeError("torchvision is required for MovingMNIST. Use --alphabet-mode instead.")  # Raise error

    cache_path = prepare_and_cache(cfg)                                              # Prepare or load cached MovingMNIST
    if cfg.prepare_only:                                                             # If only preparing data
        print("Prepared and cached dataset; exiting as requested.")                  # Inform user
        return                                                                        # Exit cleanly

    all_np = load_processed_npz(cache_path)                                          # Load cached numpy array
    N,T,H,W = all_np.shape                                                           # Unpack array shape
    n_train = min(cfg.max_train if cfg.max_train is not None else int(N*0.9), N)     # Determine training set size
    n_val = min(cfg.max_val if cfg.max_val is not None else max(1, N-n_train), N-min(n_train, N))  # Determine validation size
    if n_val == 0: n_val = max(1, N - n_train)                                       # Ensure at least one validation sequence
    train_np, val_np = all_np[:n_train], all_np[-n_val:]                             # Split arrays into train/val
    train_ds = SeqPredictDataset(train_np, cfg.t_in, cfg.t_out)                      # Wrap training dataset
    val_ds   = SeqPredictDataset(val_np,   cfg.t_in, cfg.t_out)                      # Wrap validation dataset
    pin = (device.type == "cuda")                                                    # Pin memory if using CUDA
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,  num_workers=cfg.num_workers, pin_memory=pin)  # Train loader
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)  # Val loader
    model = PixelLSTM(h=cfg.down, w=cfg.down, proj=cfg.proj, hidden=cfg.hidden).to(device)  # Build vision model
    for m in model.modules():                                                        # Iterate over submodules
        if isinstance(m, nn.Linear):                                                 # For linear layers
            nn.init.xavier_uniform_(m.weight)                                        # Xavier uniform init for weights
            if m.bias is not None: nn.init.zeros_(m.bias)                            # Zero-initialise biases
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)                            # Create Adam optimiser
    outdir = ensure_outdir(cfg.outdir)                                               # Create output directory
    weights_dir = os.path.join(cfg.weights_dir, os.path.basename(outdir))            # Mirror structure for weights
    os.makedirs(weights_dir, exist_ok=True)                                          # Ensure weights dir exists
    train_hist: List[float] = []                                                     # List for training losses
    val_hist:   List[float] = []                                                     # List for validation losses
    base_val = evaluate(model, val_loader, device)                                   # Compute baseline (untrained) loss
    train_hist.append(base_val); val_hist.append(base_val)                           # Seed histories with baseline
    if cfg.resume_from and os.path.isfile(cfg.resume_from):                          # If resuming from checkpoint
        print(f"Resuming from checkpoint: {cfg.resume_from}")                        # Inform user
        th, vh = load_checkpoint(model, cfg.resume_from)                             # Load weights and previous histories
        train_hist.extend(th); val_hist.extend(vh)                                   # Append histories for continuity
    for epoch in range(1, cfg.epochs + 1):                                           # Training loop over epochs
        t0 = time.time()                                                             # Start timer
        tr, g = train_one_epoch(model, opt, train_loader, device)                    # Train for one epoch
        vl = evaluate(model, val_loader, device)                                     # Evaluate on validation set
        train_hist.append(tr); val_hist.append(vl)                                   # Record losses
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train MSE {tr:.5f} | val MSE {vl:.5f} | grad {g:.3f} | {time.time()-t0:.1f}s")  # Progress log
    ckpt_path = save_checkpoint(model, weights_dir, train_hist, val_hist)            # Save trained model and histories
    loss_gif = os.path.join(outdir, "loss_animated.gif")                             # Path for loss animation GIF
    animate_losses(train_hist, val_hist, loss_gif)                                   # Create and save loss GIF
    with torch.no_grad():                                                            # No gradients for visual prediction
        vis_x, vis_y = val_ds[0]                                                     # Take first validation sample
        vis_x = vis_x.unsqueeze(0).to(device)                                        # Add batch dimension to inputs
        vis_y = vis_y.unsqueeze(0).to(device)                                        # Add batch dimension to targets
        preds = model.predict_future(vis_x, steps=vis_y.shape[1])                    # Predict future frames autoregressively
    gt = vis_y.squeeze(0).squeeze(1).cpu().numpy()                                   # Convert ground-truth to numpy (T,H,W)
    pr = preds.squeeze(0).squeeze(1).cpu().numpy()                                   # Convert predictions to numpy (T,H,W)
    trip_gif = os.path.join(outdir, "prediction_vs_gt.gif")                          # Path for triptych GIF
    prediction_triptych_gif(gt, pr, trip_gif)                                        # Create and save triptych GIF
    print("\n=== Summary ===")                                                       # Summary header
    print(f"Outdir:            {outdir}")                                            # Output directory path
    print(f"Weights dir:       {weights_dir}")                                       # Weights directory path
    print(f"Checkpoint:        {ckpt_path}")                                         # Checkpoint path
    print(f"Loss GIF:          {loss_gif}")                                          # Loss GIF path
    print(f"Triptych GIF:      {trip_gif}")                                          # Triptych GIF path
    if cfg.auto_open:                                                                # If user requested auto-open
        maybe_open(loss_gif); maybe_open(trip_gif)                                   # Attempt to open both GIFs

if __name__ == "__main__":                                                           # Standard Python entry-point guard
    main()                                                                            # Invoke main function
