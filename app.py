
import torch
import torch.nn as nn
import streamlit as st
from torchtext.data.utils import get_tokenizer
from collections import OrderedDict

# Define LSTM Cell
class LSTM_cell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.U_i = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_i = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))
        self.U_f = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_f = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))
        self.U_g = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_g = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(hidden_dim))
        self.U_o = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_o = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / (self.hidden_dim ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        bs, seq_len, _ = x.size()
        output = []
        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_dim).to(x.device)
            c_t = torch.zeros(bs, self.hidden_dim).to(x.device)
        else:
            h_t, c_t = init_states
        for t in range(seq_len):
            x_t = x[:, t, :]
            f_t = torch.sigmoid(h_t @ self.W_f + x_t @ self.U_f + self.b_f)
            i_t = torch.sigmoid(h_t @ self.W_i + x_t @ self.U_i + self.b_i)
            o_t = torch.sigmoid(h_t @ self.W_o + x_t @ self.U_o + self.b_o)
            g_t = torch.tanh(h_t @ self.W_g + x_t @ self.U_g + self.b_g)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            output.append(h_t.unsqueeze(0))
        output = torch.cat(output, dim=0).transpose(0, 1).contiguous()
        return output, (h_t, c_t)

# Define Custom LSTM Language Model
class CustomLSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm_cells = nn.ModuleList(
            [LSTM_cell(emb_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output = embedded
        new_hidden = []
        for i, lstm_cell in enumerate(self.lstm_cells):
            output, (h, c) = lstm_cell(output, hidden[i] if hidden else None)
            new_hidden.append((h, c))
        output = self.fc(output)
        return output, new_hidden

    def init_hidden(self, batch_size):
        return [(torch.zeros(batch_size, lstm.hidden_dim), torch.zeros(batch_size, lstm.hidden_dim)) for lstm in self.lstm_cells]

# Define the Text Generation Function
def generate(model, tokenizer, vocab, prompt, max_len=50, temperature=1.0):
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[token] if token in vocab else vocab['<unk>'] for token in tokens]
    input_seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(torch.device("cpu"))
    hidden = model.init_hidden(1)

    # Start the output with the input prompt
    generated = tokens[:]

    with torch.no_grad():
        for _ in range(max_len):
            outputs, hidden = model(input_seq, hidden)
            logits = outputs[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Break if <eos> token is generated
            if next_token == vocab['<eos>']:
                break

            # Append the generated token to the sequence
            generated.append(vocab.lookup_token(next_token))
            input_seq = torch.tensor([[next_token]], dtype=torch.long).to(torch.device("cpu"))

    return " ".join(generated)

# Load Model and Vocabulary
@st.cache_resource
def load_model():
    vocab_size = 66059  # Replace with your actual vocab size
    emb_dim = 256
    hidden_dim = 256
    num_layers = 2
    dropout = 0.5
    model = CustomLSTMLanguageModel(vocab_size, emb_dim, hidden_dim, num_layers, dropout)
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_vocab_and_tokenizer():
    vocab = torch.load("vocab.pth")  # Load your saved vocabulary
    tokenizer = get_tokenizer("basic_english")
    return vocab, tokenizer

model = load_model()
vocab, tokenizer = load_vocab_and_tokenizer()

# Streamlit App
st.title("🌟 LSTM Text Generation App 🌟")
st.markdown("Enter a prompt and let the AI generate the rest of the story!")

prompt = st.text_input("Enter your prompt here:", "Once upon a time")
max_seq_len = st.slider("Select the maximum sequence length:", 10, 100, 30, step=5)
randomness = st.slider("Set the creativity level (temperature):", 0.5, 2.0, 1.0, step=0.1)

if st.button("Generate Text"):
    generated_text = generate(model, tokenizer, vocab, prompt, max_seq_len, randomness)
    st.text_area("Generated Text:", generated_text, height=200)
