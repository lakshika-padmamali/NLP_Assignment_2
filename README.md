# NLP_Assignment_2

# Custom LSTM Text Generation

# Overview

This project implements a custom LSTM-based text generation model. Users can provide a starting prompt, and the trained model generates text interactively through a Streamlit app. The streamlit app is available in https://nlpassignment2-bfld22dt85tflf8sqsvtfq.streamlit.app/


# Files in the Repository

NLP_Ass2_st124872.ipynb: Notebook for training the custom LSTM model and saving the trained model as best_model.pth and vocabulary as vocab.pth.

best_model.pth: Saved weights of the trained LSTM model.

vocab.pth: Vocabulary file with token-to-index mappings.

app.py: Streamlit app for text generation.


# Example
Input:
Prompt: Once upon a time

Output:
Once upon a time, there was a magical land full of wonder...

# Key Features

Custom LSTM: Built from scratch using LSTM_cell and CustomLSTMLanguageModel.

Interactive App: User-friendly Streamlit app for real-time text generation
