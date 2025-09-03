# QUT_ENN572_LSTM

MovingMNIST & Alphabet LSTM — Toy Example

QUT Lecture: Artificial Intelligence in Transport
A single-file, lecture-ready demo (moving_mnist_lstm_all_in_one.py) showcasing sequence-to-sequence learning with LSTMs for two toy tasks:
Vision seq2seq: next-frame prediction on MovingMNIST
Text seq2seq: alphabet wrap-around (e.g., ABC→DEF, XYZ→YZA, A→B, E→F) with an optional Tkinter GUI
The script is self-contained, reproducible, and designed for quick classroom demos and student exploration.

![seq_00_gt_pred_err](https://github.com/user-attachments/assets/35468a11-3d0d-4dd2-9d26-8d6ebf127bf8)


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
