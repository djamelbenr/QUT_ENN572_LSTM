# QUT_ENN572_LSTM

**MovingMNIST & Alphabet LSTM — Toy Examples**  
_QUT Lecture: Artificial Intelligence in Transport_

This repository provides two compact exercises demonstrating sequence modelling with LSTMs: a vision task (MovingMNIST next-frame prediction) and a simple NLP task (alphabet sequence mapping). A second script shows a word-level LSTM language model for next-sentence completion using Project Gutenberg texts.

![seq_00_gt_pred_err](https://github.com/user-attachments/assets/35468a11-3d0d-4dd2-9d26-8d6ebf127bf8)

---

## Contents

- [Exercise 1 — `moving_mnist_lstm_all_in_one.py`](#exercise-1--moving_mnist_lstm_all_in_onepy)
- [Exercise 2 — `lstm_text_completion.py`](#exercise-2--lstm_text_completionpy)
- [Requirements](#requirements)
- [Installation](#installation)
- [Headless/Remote Environments](#headlessremote-environments)
- [Repository Structure (suggested)](#repository-structure-suggested)
- [Licence](#licence)

---

## Exercise 1 — `moving_mnist_lstm_all_in_one.py`

A lecture-ready, single-file demo of sequence-to-sequence learning with LSTMs.

**Includes**
- **Vision (seq2seq):** next-frame prediction on MovingMNIST (digit sprites).
- **Text (seq2seq):** alphabet wrap-around, e.g. `ABC → DEF`, `XYZ → YZA`, `A → B`, `E → F`.
- **Interactive testing:** optional Tkinter GUI for the alphabet task.
