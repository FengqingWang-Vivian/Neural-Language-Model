# PyTorch RNN for Word Complexity and City-Country Classification

## Introduction

This project demonstrates the power of Recurrent Neural Networks (RNNs) in natural language processing tasks using PyTorch. It tackles two distinct classification challenges:

1. **Word Complexity Classification**: An RNN model that learns to distinguish between simple and complex English words based on their character-level patterns. This can be valuable for educational applications, content adaptation, and readability assessment.

2. **City-Country Classification**: A multi-class RNN model that predicts the country of origin for given city names. This showcases how neural networks can learn subtle linguistic patterns and cultural naming conventions across different regions.

The project implements character-level processing, where each input (word or city name) is broken down into individual characters. This approach allows the model to:
- Learn morphological patterns and word structures
- Handle variable-length inputs naturally
- Capture character-level dependencies and patterns
- Work with unseen words/cities through character-level generalization

Key technical features:
- Uses GRU (Gated Recurrent Unit) for efficient sequence modeling
- Implements custom dataset preprocessing with character-level one-hot encoding
- Supports GPU acceleration for faster training
- Includes comprehensive hyperparameter tuning experiments
- Provides visualization tools for training progress and model performance

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



