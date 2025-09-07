# lstm_text_completion_gui.py
# ------------------------------------------------------------
# Simple GUI (Tkinter) to load a trained LSTM language model
# checkpoint and generate next sentence(s) for a given prompt.
#
# Compatible with the training checkpoints produced by:
#   lstm_text_completion.py
#
# Features:
# - Load checkpoint (*.pt)
# - Enter prompt text
# - Choose number of sentences to generate
# - Temperature, top-k, top-p sampling
# - Lowercase toggle (to match training tokenisation)
# ------------------------------------------------------------

import re
import math
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

# -----------------------------
# Special tokens & tokeniser
# -----------------------------
SPECIALS = {
    "PAD": "<pad>",
    "UNK": "<unk>",
    "BOS": "<bos>",
    "EOS": "<eos>",
}
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(text: str, lower: bool = True):
    return re.findall(TOKEN_RE, text.lower() if lower else text)

# -----------------------------
# Model (must match training)
# -----------------------------
class LSTMLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, emb: int = 256, hidden: int = 512, layers: int = 2,
                 dropout: float = 0.0, tie_weights: bool = True):
        super().__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.emb = torch.nn.Embedding(vocab_size, emb)
        self.lstm = torch.nn.LSTM(emb, hidden, num_layers=layers,
                                  dropout=dropout if layers > 1 else 0,
                                  batch_first=True)
        self.fc = torch.nn.Linear(hidden, vocab_size)
        self.tied = tie_weights
        self.proj = None
        if tie_weights:
            if hidden == emb:
                self.fc.weight = self.emb.weight
            else:
                self.proj = torch.nn.Linear(hidden, emb, bias=False)

    def forward(self, x, hidden=None):
        e = self.drop(self.emb(x))
        out, hidden = self.lstm(e, hidden)
        out = self.drop(out)
        if self.tied and self.proj is not None:
            out = self.proj(out)
            logits = out @ self.emb.weight.T
        else:
            logits = self.fc(out)
        return logits, hidden

# -----------------------------
# Sampling helpers
# -----------------------------
def sample_next(probs: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    probs = probs.clone()

    # Top-k
    if top_k and top_k > 0:
        vals, idx = torch.topk(probs, k=min(top_k, probs.numel()))
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[idx] = False
        probs[mask] = 0
        s = probs.sum()
        if s.item() > 0:
            probs /= s

    # Nucleus (top-p)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum > top_p
        mask[0] = False  # keep at least one
        sorted_probs[mask] = 0
        s = sorted_probs.sum()
        if s.item() > 0:
            sorted_probs /= s
        probs = torch.zeros_like(probs)
        probs[sorted_idx] = sorted_probs

    dist = torch.distributions.Categorical(probs=probs)
    return dist.sample().item()

def detok(words):
    out = []
    for i, w in enumerate(words):
        if i > 0 and re.match(r"[^\w\s]", w):
            out[-1] = out[-1] + w
        else:
            out.append(w)
    return " ".join(out)

@torch.inference_mode()
def generate_sentences(model, stoi, itos, prompt: str, device="cpu",
                       max_new_tokens=200, temperature=1.0, top_k=0, top_p=0.9,
                       num_sentences=1, lower=True) -> str:
    model.eval()

    bos_id = stoi[SPECIALS["BOS"]]
    eos_id = stoi[SPECIALS["EOS"]]
    unk_id = stoi[SPECIALS["UNK"]]

    toks = tokenize(prompt, lower=lower)
    x = torch.tensor([bos_id] + [stoi.get(t, unk_id) for t in toks],
                     dtype=torch.long, device=device).unsqueeze(0)

    generated = toks[:]
    hidden = None
    sentences_done = 0

    for _ in range(max_new_tokens):
        logits, hidden = model(x[:, -1:], hidden)   # step-by-step
        logits = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        nxt = sample_next(probs, top_k=top_k, top_p=top_p)
        word = itos[nxt]

        if word == SPECIALS["EOS"]:
            sentences_done += 1
            if sentences_done >= num_sentences:
                break
            else:
                generated.append(".")  # pleasant separator
        elif word not in SPECIALS.values():
            generated.append(word)

        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)

    return detok(generated)

# -----------------------------
# Checkpoint utilities
# -----------------------------
def infer_arch_from_state(state_dict):
    """
    Recover (emb, hidden, layers) from the saved weights.
    """
    # Embedding dim
    emb_w = state_dict.get("emb.weight")
    if emb_w is None:
        raise ValueError("Checkpoint missing 'emb.weight'")
    emb_dim = emb_w.shape[1]

    # Hidden dim from LSTM weights
    w_ih_l0 = state_dict.get("lstm.weight_ih_l0")
    if w_ih_l0 is None:
        raise ValueError("Checkpoint missing 'lstm.weight_ih_l0'")
    hidden_dim = w_ih_l0.shape[0] // 4  # (4*hidden, emb)

    # Count layers
    layers = 0
    while f"lstm.weight_ih_l{layers}" in state_dict:
        layers += 1

    return emb_dim, hidden_dim, layers

def build_model_from_ckpt(ckpt, device):
    vocab = ckpt.get("vocab", {})
    itos = vocab.get("itos", None)
    stoi = vocab.get("stoi", None)
    if not itos or not stoi:
        raise ValueError("Checkpoint does not contain vocabulary ('vocab.itos'/'vocab.stoi'). "
                         "Train with the provided script so vocab is saved.")

    state = ckpt["model"]
    emb, hidden, layers = infer_arch_from_state(state)
    vocab_size = len(itos)

    # Use dropout=0 for inference; tie_weights True (handles both tied & projected cases)
    model = LSTMLanguageModel(vocab_size, emb=emb, hidden=hidden, layers=layers,
                              dropout=0.0, tie_weights=True).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, stoi, itos

# -----------------------------
# Tkinter GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSTM Next-Sentence Predictor")
        self.geometry("980x600")

        # State
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.stoi = None
        self.itos = None
        self.ckpt_path = None

        # Top bar: Load + status
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        self.btn_load = ttk.Button(top, text="Load Model…", command=self.load_model)
        self.btn_load.pack(side="left")

        self.model_label = ttk.Label(top, text="No model loaded", foreground="#555")
        self.model_label.pack(side="left", padx=10)

        self.device_label = ttk.Label(top, text=f"Device: {self.device}")
        self.device_label.pack(side="right")

        # Middle: Prompts and Output
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Prompt box
        prompt_frame = ttk.LabelFrame(mid, text="Prompt")
        prompt_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.txt_prompt = ScrolledText(prompt_frame, wrap="word", height=18)
        self.txt_prompt.pack(fill="both", expand=True, padx=6, pady=6)
        self.txt_prompt.insert("1.0", "It was a quiet evening in the village")

        # Output box
        out_frame = ttk.LabelFrame(mid, text="Completion")
        out_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.txt_out = ScrolledText(out_frame, wrap="word", height=18)
        self.txt_out.pack(fill="both", expand=True, padx=6, pady=6)

        # Bottom: Controls
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(bottom, text="Sentences:").grid(row=0, column=0, sticky="w")
        self.var_sentences = tk.IntVar(value=1)
        ttk.Spinbox(bottom, from_=1, to=5, textvariable=self.var_sentences, width=5).grid(row=0, column=1, padx=6)

        ttk.Label(bottom, text="Temperature:").grid(row=0, column=2, sticky="w")
        self.var_temp = tk.DoubleVar(value=1.0)
        ttk.Entry(bottom, textvariable=self.var_temp, width=6).grid(row=0, column=3, padx=6)

        ttk.Label(bottom, text="top-k:").grid(row=0, column=4, sticky="w")
        self.var_topk = tk.IntVar(value=0)
        ttk.Entry(bottom, textvariable=self.var_topk, width=6).grid(row=0, column=5, padx=6)

        ttk.Label(bottom, text="top-p:").grid(row=0, column=6, sticky="w")
        self.var_topp = tk.DoubleVar(value=0.9)
        ttk.Entry(bottom, textvariable=self.var_topp, width=6).grid(row=0, column=7, padx=6)

        self.var_lower = tk.BooleanVar(value=True)
        ttk.Checkbutton(bottom, text="Lowercase input", variable=self.var_lower).grid(row=0, column=8, padx=10)

        self.btn_gen = ttk.Button(bottom, text="Generate", command=self.generate)
        self.btn_gen.grid(row=0, column=9, padx=10)

        for i in range(10):
            bottom.grid_columnconfigure(i, weight=0)
        bottom.grid_columnconfigure(10, weight=1)

        # Styling (nice, neutral)
        try:
            self.call("tk", "scaling", 1.2)
        except Exception:
            pass

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Select LSTM checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            ckpt = torch.load(path, map_location=self.device)
            model, stoi, itos = build_model_from_ckpt(ckpt, self.device)
            self.model, self.stoi, self.itos = model, stoi, itos
            self.ckpt_path = path
            self.model_label.config(text=f"Loaded: {path}")
            messagebox.showinfo("Model loaded", "Checkpoint loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def generate(self):
        if self.model is None:
            messagebox.showwarning("No model", "Please load a trained checkpoint first.")
            return

        prompt = self.txt_prompt.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("Empty prompt", "Please enter a prompt sentence.")
            return

        try:
            sentences = int(self.var_sentences.get())
            temperature = float(self.var_temp.get())
            top_k = int(self.var_topk.get())
            top_p = float(self.var_topp.get())
            lower = bool(self.var_lower.get())
        except Exception:
            messagebox.showwarning("Invalid settings", "Please check the sampling settings.")
            return

        try:
            text = generate_sentences(
                self.model, self.stoi, self.itos,
                prompt,
                device=self.device,
                max_new_tokens=200,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_sentences=sentences,
                lower=lower
            )
            self.txt_out.delete("1.0", "end")
            self.txt_out.insert("1.0", text)
        except Exception as e:
            messagebox.showerror("Generation error", f"An error occurred:\n{e}")

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
