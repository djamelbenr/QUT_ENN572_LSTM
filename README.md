# QUT_ENN572_LSTM

MovingMNIST & Alphabet LSTM — Toy Example

QUT Lecture: Artificial Intelligence in Transport

This repository contains two short exercises using LSTM networks—a simple NLP task and a digit-sequence prediction task.

Exercise 1:  pack (single file), moving_mnist_lstm_all_in_one.py — a lecture-ready, self-contained demo of sequence-to-sequence learning with LSTMs:
Vision (seq2seq): next-frame prediction on Moving MNIST (digits).
Text (seq2seq): alphabet wrap-around (e.g., ABC→DEF, XYZ→YZA, A→B, E→F), with an optional Tkinter GUI for interactive testing.
The script is reproducible and designed for quick classroom demonstrations and student exploration.

![seq_00_gt_pred_err](https://github.com/user-attachments/assets/35468a11-3d0d-4dd2-9d26-8d6ebf127bf8)

Exercise 2:

A compact, end-to-end word-level LSTM language model that learns to complete the next sentence(s) from a prompt. It fetches public-domain texts from Project Gutenberg, builds a dataset, trains a pure LSTM (no Transformers), and then generates continuations with temperature/top-k/top-p sampling.

What it does

Downloads a small corpus: Alice in Wonderland (11), Sherlock Holmes (1661), Pride and Prejudice (1342), The Time Machine (35), Dracula (345).
Cleans Project Gutenberg headers/footers and splits into sentences.
Tokenises at the word/punctuation level; lower-cases by default; builds a vocabulary with <pad>, <unk>, <bos>, <eos>.
Streams a continuous token sequence with <eos> between sentences; trains by next-token prediction (cross-entropy).
Models with a 2-layer LSTM (configurable), dropout, optional weight-tying, and gradient clipping.
Evaluates using token-level loss and perplexity; saves checkpoints (.pt and .best.pt).
Generates sentence completions using temperature, top-k and/or top-p (nucleus) sampling until a set number of sentences are completed.

Key implementation details
Dataset: contiguous token stream (word-level) with <eos> markers; 90/10 train/validation split.
Loss/metric: cross-entropy; perplexity reported per epoch.
Model: Embedding → LSTM (layers) → Linear (or projection + tied weights when enabled).
Sampling: configurable --temperature, --top-k, --top-p; stops after --gen-sentences <eos> tokens.

Train:

python lstm_text_completion.py --books alice,sherlock,pride --epochs 5 --batch-size 64 --seq-len 64


Generate (after training):

python lstm_text_completion.py --resume lstm_lm.pt \
  --generate "She opened the door and" \
  --gen-sentences 2 --temperature 0.9 --top-p 0.9


Tip: You can also generate immediately after training without --resume (the script reloads the latest save).


Requirements

Python 3.9+
PyTorch (CPU or CUDA)
NumPy, Matplotlib, ImageIO
(Optional for MovingMNIST) torchvision (dataset download/cache)
(Optional for GUI) Tkinter (bundled with most standard Python installers)

# Minimal
pip install torch numpy matplotlib imageio

# To enable MovingMNIST task
pip install torchvision


In headless environments (e.g., remote servers), the GUI falls back to a small console REPL.
