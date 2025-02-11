README: PyTorch RNN for Word Complexity and City-Country Classification

Overview

This project implements a Recurrent Neural Network (RNN) using PyTorch to classify words into simple or hard and to predict the country of a given city. The model is trained using character-level embeddings and learns to recognize patterns in word structures.

Features
	•	Character-level embedding-based RNN for sequence modeling.
	•	Binary classification of words as simple or hard.
	•	Multi-class classification to predict the country given a city name.
	•	Uses GRU (Gated Recurrent Unit) for improved learning efficiency.
	•	Implements custom dataset preprocessing with one-hot encoding of characters.
	•	Trains on small to medium datasets with minimal hyperparameter tuning.
	•	Supports GPU acceleration for faster training.

Installation

Requirements
	•	Python (>=3.8)
	•	PyTorch (>=1.10)
	•	NumPy
	•	Pandas
	•	scikit-learn
	•	Matplotlib (for visualization)

Install Dependencies

pip install torch numpy pandas scikit-learn matplotlib

Dataset

Word Complexity Classification
	•	Input: English words (strings)
	•	Output: Binary label (0 = Simple, 1 = Hard)
	•	Example:

"apple" → 0
"ephemeral" → 1



City-Country Classification
	•	Input: City names (strings)
	•	Output: Country label (string)
	•	Example:

"Paris" → "fr"
"Beijing" → "cn"



Data Format (txt)

Both datasets should be stored in txt format:

word,label
apple,0
ephemeral,1

city,country
Paris,fr
Beijing,cn

Model Architecture

1. Character Embedding Layer

Each word/city is processed at the character level, where each character is one-hot encoded and passed through an embedding layer.

2. RNN (GRU-based)
	•	Input: Character embeddings
	•	Hidden Layer: GRU with hidden state propagation
	•	Output: Fully connected layer with softmax (multi-class) or sigmoid (binary) activation



